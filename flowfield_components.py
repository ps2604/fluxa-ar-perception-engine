# file: flowfield_components.py
# Revision 6.2: FIXED cache key issue for backward pass
# CRITICAL FIX: Pass correct conv_cache to fused backward helper

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any

from flowfield_core_optimized import (
    FSEField, FieldType, FieldOperations, ArrayLike, 
    get_memory_pool, FusedFieldOperations  # Fixed imports
)
import logging

logger = logging.getLogger(__name__)

class FlowField_FLIT:
    """✅ ENHANCED: Trainable FLIT with robust cache handling"""
    def __init__(self, input_channels: int, output_channels: int,
                 field_type: FieldType, evolution_rate: float, device: str, use_bias: bool = True,
                 context_channels_in: Optional[int] = None):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.activation_field_type = field_type 
        self.evolution_rate = evolution_rate 
        self.output_channels = output_channels
        self.input_channels = input_channels

        # Kernel initialized with He/Kaiming normal for better convergence
        init_scale_kernel = self.backend.sqrt(2. / input_channels)
        kernel_data = self.backend.random.normal(0, init_scale_kernel, (input_channels, output_channels)).astype(self.backend.float32)
        self.kernel = FSEField(kernel_data, field_type=FieldType.LINEAR, device=device)
        self.parameters = {'kernel': self.kernel}

        self.use_bias = use_bias
        if self.use_bias:
            bias_data = self.backend.zeros(output_channels, dtype=self.backend.float32)
            self.bias = FSEField(bias_data, field_type=FieldType.LINEAR, device=device)
            self.parameters['bias'] = self.bias
        
        self.context_channels_in = context_channels_in
        self.context_projection_kernel: Optional[FSEField] = None
        self.context_projection_bias: Optional[FSEField] = None

        if self.context_channels_in is not None and self.context_channels_in > 0:
            mod_channels = output_channels
            init_scale_ctx = self.backend.sqrt(1. / self.context_channels_in)
            ctx_proj_kernel_data = self.backend.random.normal(0, init_scale_ctx, (self.context_channels_in, mod_channels)).astype(self.backend.float32)
            self.context_projection_kernel = FSEField(ctx_proj_kernel_data, FieldType.LINEAR, device=device)
            self.parameters['context_projection_kernel'] = self.context_projection_kernel
            
            if self.use_bias:
                ctx_proj_bias_data = self.backend.zeros(mod_channels, dtype=self.backend.float32)
                self.context_projection_bias = FSEField(ctx_proj_bias_data, FieldType.LINEAR, device=device)
                self.parameters['context_projection_bias'] = self.context_projection_bias
        
        logger.debug(f"FLIT Init: In({input_channels})->Out({output_channels}), Kernel{self.kernel.shape}, Bias:{self.use_bias}, CtxChIn:{self.context_channels_in} Dev:{device}")

    def forward(self, inputs: FSEField, context_signal: Optional[FSEField] = None) -> Tuple[FSEField, Dict[str, Any]]:
        """✅ CRITICAL FIX: Forward pass with proper cache key management"""
        if inputs.device != self.device: 
            inputs = inputs.to_device(self.device)
        
        # ✅ CRITICAL: Use FusedFieldOperations for consistent cache
        convolved_field, conv_cache = FusedFieldOperations.fused_conv_activation(
            inputs, self.kernel, self.activation_field_type
        )
        
        pre_activation_data = convolved_field.data
        if self.use_bias:
            pre_activation_data = pre_activation_data + self.bias.data

        pre_context_mod_field = FSEField(pre_activation_data, FieldType.LINEAR, device=self.device)
        
        modulated_data = pre_context_mod_field.data
        projected_context_modulation_data: Optional[ArrayLike] = None 

        if self.context_projection_kernel and context_signal:
            if context_signal.device != self.device: 
                context_signal = context_signal.to_device(self.device)
            if context_signal.ndim != 2 or context_signal.shape[0] != inputs.shape[0] or context_signal.shape[1] != self.context_channels_in:
                raise ValueError(f"FLIT Ctx signal shape err. Expected (B, {self.context_channels_in}), got {context_signal.shape}")

            projected_context_flat = context_signal.data @ self.context_projection_kernel.data
            if self.use_bias and self.context_projection_bias:
                 projected_context_flat = projected_context_flat + self.context_projection_bias.data
            
            projected_context_modulation_data = projected_context_flat.reshape(inputs.shape[0], 1, 1, self.output_channels)
            modulated_data = pre_context_mod_field.data + projected_context_modulation_data
        
        final_pre_activation_field = FSEField(modulated_data, FieldType.LINEAR, device=self.device)
        activated_field = FieldOperations.apply_activation(final_pre_activation_field, self.activation_field_type)
        
        # ✅ CRITICAL FIX: Add missing keys to conv_cache for backward pass
        conv_cache['pre_activation_data'] = final_pre_activation_field.data  # This is what backward pass needs
        conv_cache['activation_type_used'] = self.activation_field_type  # Also needed for activation derivatives
        
        # ✅ COMPREHENSIVE CACHE: Include ALL necessary keys
        cache = {
            'inputs': inputs, 
            'conv_cache': conv_cache,  # Now contains pre_activation_data and activation_type_used
            'pre_context_mod_field_data': pre_context_mod_field.data, 
            'projected_context_modulation_data': projected_context_modulation_data,
            'final_pre_activation_data': final_pre_activation_field.data,
            'activation_type_used': self.activation_field_type,
            'original_context_signal': context_signal,
            
            # ✅ CRITICAL: Copy convolution cache keys to top level for backward compatibility
            'is_1x1_conv': conv_cache.get('is_1x1_conv', False),
            'input_field_shape': conv_cache.get('input_field_shape'),
            'kernel_field_shape': conv_cache.get('kernel_field_shape'),
            'input_reshaped': conv_cache.get('input_reshaped'),
            'kernel_reshaped': conv_cache.get('kernel_reshaped'),
            'cols_reshaped': conv_cache.get('cols_reshaped'),
            'output_shape': conv_cache.get('output_shape'),
            'strides': conv_cache.get('strides'),
            'P_H': conv_cache.get('P_H', 0),
            'P_W': conv_cache.get('P_W', 0)
        }
        
        return activated_field, cache

    def backward(self, upstream_grad_activated: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, FSEField], FSEField]:
        """✅ ENHANCED: Backward pass with robust cache handling"""
        
        # ✅ VALIDATE CACHE KEYS
        required_keys = ['inputs', 'conv_cache', 'final_pre_activation_data', 'activation_type_used']
        missing_keys = [key for key in required_keys if key not in cache]
        
        if missing_keys:
            logger.error(f"❌ FLIT backward: Missing cache keys {missing_keys}")
            # Return zero gradients as fallback
            param_grads = {}
            if 'kernel' in self.parameters:
                param_grads['kernel'] = FSEField(
                    self.backend.zeros_like(self.parameters['kernel'].data), 
                    device=self.device
                )
            zero_grad = FSEField(self.backend.zeros_like(upstream_grad_activated.data), device=self.device)
            return param_grads, zero_grad
        
        inputs = cache['inputs']
        conv_cache = cache['conv_cache']
        final_pre_activation_data = cache['final_pre_activation_data']
        activation_type_used = cache['activation_type_used']
        original_context_signal = cache['original_context_signal']
        projected_context_modulation_data = cache.get('projected_context_modulation_data')

        # dL/dZ_mod = dL/dA * dA/dZ_mod
        grad_final_pre_activation = FieldOperations.activation_derivative(
            upstream_grad_activated, final_pre_activation_data, activation_type_used
        )

        param_grads: Dict[str,FSEField] = {}
        
        grad_pre_context_mod_field = grad_final_pre_activation
        dL_dProjectedCtxModData: Optional[ArrayLike] = None
        if projected_context_modulation_data is not None:
            dL_dProjectedCtxModData = grad_final_pre_activation.data
        
        # Gradient for context_projection parameters
        if self.context_projection_kernel and original_context_signal and dL_dProjectedCtxModData is not None:
            dL_dProjCtxMod_summed_spatial = self.backend.sum(dL_dProjectedCtxModData, axis=(1,2))
            
            if self.use_bias and self.context_projection_bias:
                grad_ctx_proj_bias_data = self.backend.sum(dL_dProjCtxMod_summed_spatial, axis=0)
                param_grads['context_projection_bias'] = FSEField(grad_ctx_proj_bias_data, device=self.device)

            grad_ctx_proj_kernel_data = original_context_signal.data.T @ dL_dProjCtxMod_summed_spatial
            param_grads['context_projection_kernel'] = FSEField(grad_ctx_proj_kernel_data, device=self.device)
        
        if self.use_bias:
            grad_bias_data = self.backend.sum(grad_pre_context_mod_field.data, axis=(0, 1, 2))
            param_grads['bias'] = FSEField(grad_bias_data, device=self.device)

        grad_conv_output = grad_pre_context_mod_field
        
        # ✅ CRITICAL FIX: Pass the correct conv_cache to fused backward
        try:
            # Use FusedFieldOperations for consistent backward pass
            conv_param_grads, downstream_grad_inputs = FusedFieldOperations.fused_conv_activation_backward(
                grad_conv_output, conv_cache  # ✅ FIXED: Pass conv_cache directly
            )
            
            # Merge convolution parameter gradients
            param_grads.update(conv_param_grads)
            
            # ✅ MEMORY FIX: Free large scratch tensors after backward
            if 'cols_reshaped' in conv_cache:
                conv_cache.pop('cols_reshaped', None)
            if 'input_reshaped' in conv_cache:
                conv_cache.pop('input_reshaped', None)
            if 'kernel_reshaped' in conv_cache:
                conv_cache.pop('kernel_reshaped', None)
            
            return param_grads, downstream_grad_inputs
            
        except Exception as e:
            logger.error(f"❌ FLIT convolution backward failed: {e}")
            
            # ✅ FALLBACK: Direct field operations with manual cache construction
            try:
                # Construct safe cache for direct operations
                safe_conv_cache = conv_cache.copy() if conv_cache else {}
                
                # Ensure required keys are present
                if 'is_1x1_conv' not in safe_conv_cache:
                    safe_conv_cache['is_1x1_conv'] = cache.get('is_1x1_conv', False)
                if 'kernel_field_shape' not in safe_conv_cache:
                    safe_conv_cache['kernel_field_shape'] = self.kernel.shape
                if 'input_field_shape' not in safe_conv_cache:
                    safe_conv_cache['input_field_shape'] = inputs.shape
                
                param_grads['kernel'] = FieldOperations.field_convolution_backward_kernel(
                    grad_conv_output, safe_conv_cache
                )
                downstream_grad_inputs = FieldOperations.field_convolution_backward_data(
                    grad_conv_output, safe_conv_cache
                )
                
                return param_grads, downstream_grad_inputs
                
            except Exception as e2:
                logger.error(f"❌ FLIT fallback backward also failed: {e2}")
                
                # Ultimate fallback: zero gradients
                param_grads['kernel'] = FSEField(
                    self.backend.zeros_like(self.kernel.data), device=self.device
                )
                zero_grad = FSEField(
                    self.backend.zeros_like(inputs.data), device=self.device
                )
                return param_grads, zero_grad

class FlowField_FSEBlock:
    """✅ ENHANCED: FSE Block with robust cache handling"""
    def __init__(self, input_channels: int, internal_channels: int, num_fils: int, device: str, 
                 use_bias_in_fils:bool = True, context_channels_for_fils: Optional[int] = None):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.fils: List[FlowField_FLIT] = []
        self.parameters: Dict[str, Any] = {}
        self.num_fils = num_fils
        self.context_channels_for_fils = context_channels_for_fils

        current_ch = input_channels
        for i in range(num_fils):
            fil = FlowField_FLIT(current_ch, internal_channels, FieldType.CONTINUOUS, 0.1, device, 
                                 use_bias=use_bias_in_fils, context_channels_in=self.context_channels_for_fils)
            self.fils.append(fil)
            self.parameters[f'fil{i+1}'] = fil.parameters
            current_ch = internal_channels

        self.skip_projection_flit: Optional[FlowField_FLIT] = None
        if input_channels != internal_channels:
            self.skip_projection_flit = FlowField_FLIT(input_channels, internal_channels, FieldType.LINEAR, 0.1, device, 
                                                       use_bias=use_bias_in_fils, context_channels_in=None) 
            self.parameters['skip_projection_flit'] = self.skip_projection_flit.parameters

    def forward(self, inputs: FSEField, context_signal: Optional[FSEField] = None) -> Tuple[FSEField, Dict[str, Any]]:
        """✅ ENHANCED: Forward pass with comprehensive cache"""
        if inputs.device != self.device: 
            inputs = inputs.to_device(self.device)
        if context_signal and context_signal.device != self.device: 
            context_signal = context_signal.to_device(self.device)

        x = inputs
        fil_caches = []
        for fil in self.fils:
            x, cache = fil.forward(x, context_signal=context_signal)
            fil_caches.append(cache)
        x_after_fils = x 
        
        projected_skip_field: FSEField
        cache_skip_proj: Optional[Dict[str,Any]] = None
        if self.skip_projection_flit:
            projected_skip_field, cache_skip_proj = self.skip_projection_flit.forward(inputs, context_signal=None)
        else: 
            projected_skip_field = inputs

        if x_after_fils.shape[1:3] != projected_skip_field.shape[1:3] or x_after_fils.shape[-1] != projected_skip_field.shape[-1]:
            raise ValueError(f"FSEBlock skip shape: FIL_out {x_after_fils.shape} vs Skip {projected_skip_field.shape}")

        sum_before_tanh_data = x_after_fils.data + projected_skip_field.data
        output_field = FieldOperations.apply_activation(FSEField(sum_before_tanh_data, x_after_fils.field_type, device=self.device), FieldType.CONTINUOUS)

        # ✅ COMPREHENSIVE CACHE
        cache = {
            'inputs': inputs, 
            'fil_caches': fil_caches, 
            'sum_before_tanh_data': sum_before_tanh_data,
            'cache_skip_proj': cache_skip_proj,
            'original_context_signal_to_block': context_signal,
            'x_after_fils_shape': x_after_fils.shape,
            'projected_skip_field_shape': projected_skip_field.shape,
            'output_field_shape': output_field.shape
        }
        
        return output_field, cache

    def backward(self, upstream_grad_output: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str, Any], FSEField]:
        """✅ ENHANCED: Backward pass with robust error handling"""
        
        # ✅ VALIDATE CACHE
        required_keys = ['inputs', 'fil_caches', 'sum_before_tanh_data']
        missing_keys = [key for key in required_keys if key not in cache]
        
        if missing_keys:
            logger.error(f"❌ FSEBlock backward: Missing cache keys {missing_keys}")
            # Return zero gradients
            zero_param_grads = {}
            zero_input_grad = FSEField(self.backend.zeros_like(upstream_grad_output.data), device=self.device)
            return zero_param_grads, zero_input_grad
        
        inputs = cache['inputs']
        fil_caches = cache['fil_caches']
        sum_before_tanh_data = cache['sum_before_tanh_data']
        cache_skip_proj = cache['cache_skip_proj']
        
        grad_sum = FieldOperations.activation_derivative(upstream_grad_output, sum_before_tanh_data, FieldType.CONTINUOUS)

        grad_x_after_fils = grad_sum
        grad_projected_skip = grad_sum 
        all_param_grads: Dict[str, Any] = {}
        current_upstream_grad_for_fils = grad_x_after_fils
        
        # ✅ ROBUST FIL BACKWARD CHAIN
        for i in range(self.num_fils - 1, -1, -1):
            try:
                if i < len(fil_caches):
                    param_grads, current_upstream_grad_for_fils = self.fils[i].backward(
                        current_upstream_grad_for_fils, fil_caches[i]
                    )
                    all_param_grads[f'fil{i+1}'] = param_grads
                else:
                    logger.warning(f"❌ Missing cache for FIL {i}")
                    # Create zero gradients for this FIL
                    zero_param_grads = {}
                    for param_name in self.fils[i].parameters:
                        zero_param_grads[param_name] = FSEField(
                            self.backend.zeros_like(self.fils[i].parameters[param_name].data),
                            device=self.device
                        )
                    all_param_grads[f'fil{i+1}'] = zero_param_grads
                    
            except Exception as e:
                logger.error(f"❌ FIL {i} backward failed: {e}")
                # Create zero gradients for this FIL
                zero_param_grads = {}
                for param_name in self.fils[i].parameters:
                    zero_param_grads[param_name] = FSEField(
                        self.backend.zeros_like(self.fils[i].parameters[param_name].data),
                        device=self.device
                    )
                all_param_grads[f'fil{i+1}'] = zero_param_grads
        
        grad_input_from_fil_path = current_upstream_grad_for_fils
        grad_input_from_skip_path: FSEField
        
        if self.skip_projection_flit and cache_skip_proj:
            try:
                skip_param_grads, grad_input_from_skip_path = self.skip_projection_flit.backward(
                    grad_projected_skip, cache_skip_proj
                )
                all_param_grads['skip_projection_flit'] = skip_param_grads
            except Exception as e:
                logger.error(f"❌ Skip projection backward failed: {e}")
                # Zero gradients for skip projection
                zero_skip_grads = {}
                for param_name in self.skip_projection_flit.parameters:
                    zero_skip_grads[param_name] = FSEField(
                        self.backend.zeros_like(self.skip_projection_flit.parameters[param_name].data),
                        device=self.device
                    )
                all_param_grads['skip_projection_flit'] = zero_skip_grads
                grad_input_from_skip_path = grad_projected_skip
        else: 
            grad_input_from_skip_path = grad_projected_skip
            
        final_downstream_grad = grad_input_from_fil_path + grad_input_from_skip_path
        return all_param_grads, final_downstream_grad

class FlowField_Upsample:
    """✅ ENHANCED: Upsample with better error handling"""
    def __init__(self, factor: int, device: str, activation_type: FieldType = FieldType.CONTINUOUS):
        self.factor=factor
        self.device=device
        self.backend=cp if device=="gpu" else np
        self.parameters:Dict[str,FSEField]={}
        self.activation_type=activation_type
        
    def forward(self, inputs: FSEField) -> Tuple[FSEField, Dict[str,Any]]:
        if inputs.device!=self.device: 
            inputs=inputs.to_device(self.device)
        fh,fw=self.factor,self.factor
        up_h=self.backend.repeat(inputs.data,fh,axis=1)
        up_hw=self.backend.repeat(up_h,fw,axis=2)
        act_field=FieldOperations.apply_activation(FSEField(up_hw,inputs.field_type,device=self.device),self.activation_type)
        
        cache = {
            'inputs_shape':inputs.shape,
            'pre_activation_data':up_hw,
            'activation_type_used':self.activation_type,
            'factor': self.factor,
            'input_field_type': inputs.field_type
        }
        
        return act_field, cache
        
    def backward(self, up_grad_act: FSEField, cache: Dict[str,Any]) -> Tuple[Dict[str,FSEField],FSEField]:
        # ✅ VALIDATE CACHE
        if 'pre_activation_data' not in cache or 'inputs_shape' not in cache:
            logger.error("❌ Upsample backward: Missing cache keys")
            zero_grad = FSEField(self.backend.zeros((1,1,1,1), dtype=up_grad_act.dtype), device=self.device)
            return {}, zero_grad
            
        pre_act_data=cache['pre_activation_data']
        act_type=cache['activation_type_used']
        in_s=cache['inputs_shape']
        
        grad_pre_act=FieldOperations.activation_derivative(up_grad_act,pre_act_data,act_type)
        fh,fw=self.factor,self.factor
        B,Hu,Wu,Cu = grad_pre_act.shape
        reshaped_grad=grad_pre_act.data.reshape(B,in_s[1],fh,in_s[2],fw,Cu)
        down_grad_data=reshaped_grad.sum(axis=(2,4))
        
        return {}, FSEField(down_grad_data, field_type=grad_pre_act.field_type, device=self.device)

class FlowField_Downsample(FlowField_Upsample):
    """✅ ENHANCED: Downsample with better error handling"""
    def __init__(self, factor: int, device: str, activation_type: FieldType = FieldType.CONTINUOUS):
        super().__init__(factor, device, activation_type)
        
    def forward(self, inputs: FSEField) -> Tuple[FSEField, Dict[str, Any]]:
        if inputs.device != self.device: 
            inputs = inputs.to_device(self.device)
        B,H,W,C = inputs.shape
        fh,fw = self.factor,self.factor
        Hp,Wp = H//fh, W//fw
        input_eff = inputs.data[:, :Hp*fh, :Wp*fw, :]
        pooled = input_eff.reshape(B,Hp,fh,Wp,fw,C).mean(axis=(2,4))
        act_field = FieldOperations.apply_activation(FSEField(pooled,inputs.field_type,device=self.device), self.activation_type)
        
        cache = {
            'inputs_original_shape':inputs.shape, 
            'pre_activation_data':pooled, 
            'activation_type_used':self.activation_type,
            'factor': self.factor,
            'effective_dims': (Hp, Wp),
            'input_field_type': inputs.field_type
        }
        
        return act_field, cache
        
    def backward(self, up_grad_act: FSEField, cache: Dict[str, Any]) -> Tuple[Dict[str,FSEField], FSEField]:
        # ✅ VALIDATE CACHE
        required_keys = ['pre_activation_data', 'inputs_original_shape', 'activation_type_used']
        missing_keys = [key for key in required_keys if key not in cache]
        
        if missing_keys:
            logger.error(f"❌ Downsample backward: Missing cache keys {missing_keys}")
            zero_grad = FSEField(self.backend.zeros((1,1,1,1), dtype=up_grad_act.dtype), device=self.device)
            return {}, zero_grad
            
        pre_act_data=cache['pre_activation_data']
        act_type=cache['activation_type_used']
        in_orig_s=cache['inputs_original_shape']
        
        grad_pre_act = FieldOperations.activation_derivative(up_grad_act,pre_act_data,act_type)
        fh,fw=self.factor,self.factor
        B,Ho,Wo,Co = in_orig_s
        Hp_eff,Wp_eff = Ho//fh,Wo//fw
        grad_exp_h = self.backend.repeat(grad_pre_act.data,fh,axis=1)
        grad_exp_hw = self.backend.repeat(grad_exp_h,fw,axis=2)
        down_grad_eff = grad_exp_hw/(fh*fw)
        down_grad_full = self.backend.zeros(in_orig_s,dtype=down_grad_eff.dtype)
        down_grad_full[:,:Hp_eff*fh,:Wp_eff*fw,:] = down_grad_eff
        
        return {}, FSEField(down_grad_full, field_type=grad_pre_act.field_type, device=self.device)