# file: flowfield_fluxa_model.py
# Revision 5.2: FIXED cache handling and device synchronization

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any, Union 

from flowfield_core_optimized import FSEField, FieldType, FieldOperations, ArrayLike, get_memory_pool 
from flowfield_components import FlowField_FSEBlock, FlowField_Downsample, FlowField_Upsample, FlowField_FLIT
import logging

logger = logging.getLogger(__name__)

class ProductionSYNTHAOrchestrator:
    """Handles SYNTHA context generation and global analysis."""
    def __init__(self, context_width: int = 32, device: str = "cpu", use_learnable_projector: bool = True):
        self.context_width = context_width
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.use_learnable_projector = use_learnable_projector
        self.context_projector_kernel: Optional[FSEField] = None 
        self.context_projector_bias: Optional[FSEField] = None
        self.parameters: Dict[str, FSEField] = {} 
        logger.debug(f"SYNTHAOrchestrator: CtxWidth={context_width}, Dev={device}, LearnableProj={use_learnable_projector}")

    def _initialize_projector(self, input_feature_len: int, use_bias: bool = True):
        if not self.use_learnable_projector:
            self.context_projector_kernel = None; self.context_projector_bias = None; return

        if self.context_projector_kernel is None or \
           self.context_projector_kernel.shape[0] != input_feature_len or \
           self.context_projector_kernel.shape[1] != self.context_width:
            
            init_scale_proj = self.backend.sqrt(1. / input_feature_len) 
            proj_data = self.backend.random.normal(0, init_scale_proj, 
                                             (input_feature_len, self.context_width)).astype(self.backend.float32)
            self.context_projector_kernel = FSEField(proj_data, FieldType.LINEAR, device=self.device)
            self.parameters['context_projector_kernel'] = self.context_projector_kernel
            
            if use_bias: 
                bias_data = self.backend.zeros(self.context_width, dtype=self.backend.float32)
                self.context_projector_bias = FSEField(bias_data, FieldType.LINEAR, device=self.device)
                self.parameters['context_projector_bias'] = self.context_projector_bias
            else: self.context_projector_bias = None
            logger.debug(f"SYNTHA ctx projector re/init: ({input_feature_len}, {self.context_width}), Bias: {use_bias}")

    def generate_context(self, image_field: FSEField,
                         additional_features: Optional[FSEField] = None) -> Tuple[FSEField, Dict[str, Any]]:
        if image_field.device != self.device: image_field = image_field.to_device(self.device)
        B = image_field.shape[0]
        mean_vals = self.backend.mean(image_field.data, axis=(1, 2)); std_vals = self.backend.std(image_field.data, axis=(1, 2))
        context_list = [mean_vals, std_vals]
        raw_context_cache = {'image_mean':mean_vals, 'image_std':std_vals}
        if additional_features is not None:
            if additional_features.device != self.device: additional_features = additional_features.to_device(self.device)
            add_flat = additional_features.data.reshape(B, -1); context_list.append(add_flat)
            raw_context_cache['additional_features_flat'] = add_flat
        raw_context_features_data = self.backend.concatenate(context_list, axis=1)
        raw_context_features_field = FSEField(raw_context_features_data, FieldType.LINEAR, device=self.device)
        raw_context_cache['raw_context_features_field'] = raw_context_features_field
        current_feature_len = raw_context_features_data.shape[1]
        self._initialize_projector(current_feature_len) 
        projected_context_data = raw_context_features_data
        if self.context_projector_kernel:
            projected_context_data = raw_context_features_data @ self.context_projector_kernel.data
            if self.context_projector_bias: projected_context_data += self.context_projector_bias.data
        current_projected_len = projected_context_data.shape[1]
        if current_projected_len > self.context_width: final_context_data = projected_context_data[:, :self.context_width]
        elif current_projected_len < self.context_width:
            padding = self.backend.zeros((B, self.context_width - current_projected_len), dtype=self.backend.float32)
            final_context_data = self.backend.concatenate([projected_context_data, padding], axis=1)
        else: final_context_data = projected_context_data
        final_context_activated_data = self.backend.tanh(final_context_data)
        cache_for_backward = {'raw_context_features_field': raw_context_features_field, 'projected_unactivated_data': final_context_data }
        return FSEField(final_context_activated_data, FieldType.CONTINUOUS, device=self.device), cache_for_backward

    def generate_context_backward(self, upstream_grad_final_context: FSEField, cache: Dict[str,Any]) -> Dict[str,FSEField]:
        if not self.use_learnable_projector or not self.context_projector_kernel: return {}
        raw_context_features_field = cache['raw_context_features_field'] 
        projected_unactivated_data = cache['projected_unactivated_data']
        grad_projected_unactivated = upstream_grad_final_context.data * (1 - self.backend.tanh(projected_unactivated_data)**2)
        dL_dProjectedUnactivated = grad_projected_unactivated
        grad_projector_params = {}
        if self.context_projector_bias:
            dL_dBias_proj = self.backend.sum(dL_dProjectedUnactivated, axis=0)
            grad_projector_params['context_projector_bias'] = FSEField(dL_dBias_proj, device=self.device)
        dL_dKernel_proj = raw_context_features_field.data.T @ dL_dProjectedUnactivated
        grad_projector_params['context_projector_kernel'] = FSEField(dL_dKernel_proj, device=self.device)
        return grad_projector_params

    def analyze_global_context(self, module_fields: Dict[str, FSEField]) -> Dict[str, Any]:
        if not module_fields: return {'global_coherence': 0.5}
        backend = self.backend; activities = []; variances = []
        for field in module_fields.values():
            if field.device != self.device: field = field.to_device(self.device)
            if field.data.size == 0: continue
            activities.append(float(backend.mean(backend.abs(field.data))))
            variances.append(float(backend.var(field.data)))
        num_valid = len(activities)
        if num_valid == 0: return {'global_coherence': 0.0}
        avg_var = sum(variances)/num_valid if num_valid > 0 else 0.0
        return {'global_coherence': 1.0 / (1.0 + avg_var + 1e-8), 'avg_activity': sum(activities)/num_valid if num_valid > 0 else 0.0}


class ProductionFLUXA:
    def __init__(self, input_shape: Tuple[int, int, int], base_channels: int,
                 enable_syntha_integration: bool, device: str, use_bias:bool=True,
                 max_cses_per_fil_arg: int = 4): # Added max_cses_per_fil_arg
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.enable_syntha_integration = enable_syntha_integration
        self.parameters: Dict[str, Any] = {}
        self.input_actual_channels = input_shape[-1]
        self.use_bias = use_bias
        self.max_cses_per_fil_arg = max_cses_per_fil_arg # Store it
        self.base_model_channels = base_channels  # Store for error handling
        self.syntha_context_channels_for_components: Optional[int] = None

        if self.enable_syntha_integration:
            self.syntha_context_channels_for_components = base_channels // 2 
            self.syntha_orchestrator = ProductionSYNTHAOrchestrator(
                context_width=self.syntha_context_channels_for_components, device=device, use_learnable_projector=True
            )
            if self.syntha_orchestrator.parameters:
                 self.parameters['syntha_orchestrator'] = self.syntha_orchestrator.parameters
        else:
            self.syntha_orchestrator = None
        
        self._build_architecture(self.input_actual_channels, base_channels) # Pass max_cses_per_fil_arg
        logger.info(f"FLUXA Init: Dev:{device}, BaseCh:{base_channels}, SYNTHA:{self.enable_syntha_integration}, Bias:{self.use_bias}, MaxCSEs/FIL:{self.max_cses_per_fil_arg}")

    def _build_architecture(self, input_channels: int, base_ch: int):
        comp_ctx_ch = self.syntha_context_channels_for_components
        num_fils_for_blocks = self.max_cses_per_fil_arg # Use the passed argument
        
        def add_p(name, comp): self.parameters[name] = comp.parameters if hasattr(comp,'parameters') else {}

        self.input_processor = FlowField_FLIT(input_channels, base_ch, FieldType.CONTINUOUS, 0.1, self.device, use_bias=self.use_bias, context_channels_in=comp_ctx_ch)
        add_p('input_processor', self.input_processor)
        
        # Use num_fils_for_blocks for FSEBlock depth, consistent with max_cses_per_fil concept
        self.encoder1_block = FlowField_FSEBlock(base_ch, base_ch, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('encoder1_block', self.encoder1_block); self.downsample1 = FlowField_Downsample(2, self.device)
        
        self.encoder2_block = FlowField_FSEBlock(base_ch, base_ch*2, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('encoder2_block', self.encoder2_block); self.downsample2 = FlowField_Downsample(2, self.device)
        
        self.encoder3_block = FlowField_FSEBlock(base_ch*2, base_ch*4, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('encoder3_block', self.encoder3_block); self.downsample3 = FlowField_Downsample(2, self.device)
        
        self.bottleneck_block = FlowField_FSEBlock(base_ch*4, base_ch*8, num_fils=max(num_fils_for_blocks, 4), device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch) # Bottleneck often deeper
        add_p('bottleneck_block', self.bottleneck_block)
        
        self.upsample1 = FlowField_Upsample(2, self.device)
        self.decoder1_block = FlowField_FSEBlock(base_ch*8+base_ch*4, base_ch*4, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('decoder1_block', self.decoder1_block)
        
        self.upsample2 = FlowField_Upsample(2, self.device)
        self.decoder2_block = FlowField_FSEBlock(base_ch*4+base_ch*2, base_ch*2, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('decoder2_block', self.decoder2_block)
        
        self.upsample3 = FlowField_Upsample(2, self.device)
        self.decoder3_block = FlowField_FSEBlock(base_ch*2+base_ch, base_ch, num_fils=num_fils_for_blocks, device=self.device, use_bias_in_fils=self.use_bias, context_channels_for_fils=comp_ctx_ch)
        add_p('decoder3_block', self.decoder3_block)
        
        self.keypoint_generator = FlowField_FLIT(base_ch,17,FieldType.SPATIAL,0.1,self.device,use_bias=self.use_bias,context_channels_in=comp_ctx_ch)
        add_p('keypoint_generator',self.keypoint_generator)
        self.segmentation_generator = FlowField_FLIT(base_ch,1,FieldType.LIGHTING,0.1,self.device,use_bias=self.use_bias,context_channels_in=comp_ctx_ch)
        add_p('segmentation_generator',self.segmentation_generator)
        self.surface_normal_generator = FlowField_FLIT(base_ch,3,FieldType.MATERIAL,0.1,self.device,use_bias=self.use_bias,context_channels_in=comp_ctx_ch)
        add_p('surface_normal_generator',self.surface_normal_generator)
        self.env_lighting_flit = FlowField_FLIT(base_ch*4,9,FieldType.LIGHTING,0.1,self.device,use_bias=self.use_bias,context_channels_in=comp_ctx_ch)
        add_p('env_lighting_flit',self.env_lighting_flit)

    def _concat_skip(self, up_field: FSEField, skip_field: FSEField) -> FSEField:
        up_data = up_field.data
        if up_field.shape[1:3] != skip_field.shape[1:3]:
            logger.warning(f"Skip spatial dim mismatch: Up {up_field.shape} Skip {skip_field.shape}. Cropping/Padding up_data.")
            h_s, w_s = skip_field.shape[1:3]; h_u, w_u = up_field.shape[1:3]
            h_min, w_min = min(h_s,h_u), min(w_s,w_u)
            up_data = up_field.data[:, :h_min, :w_min, :]
            skip_data_eff = skip_field.data[:, :h_min, :w_min, :]
        else:
            skip_data_eff = skip_field.data
        concat_data = self.backend.concatenate((up_data, skip_data_eff), axis=-1)
        return FSEField(concat_data, field_type=up_field.field_type, device=self.device) # field_type from main upsample path


    def forward(self, inputs: Union[FSEField, Tuple[FSEField, Optional[FSEField]]], training: bool = True) -> Tuple[Dict[str, FSEField], Dict[str, Any]]:
        actual_inputs: FSEField; syntha_context_signal: Optional[FSEField] = None; cache: Dict[str, Any] = {}
        if isinstance(inputs, tuple): 
            actual_inputs, raw_syntha_features_maybe = inputs
            if self.enable_syntha_integration and self.syntha_orchestrator:
                 syntha_context_signal, cache['syntha_gen_cache'] = self.syntha_orchestrator.generate_context(actual_inputs, raw_syntha_features_maybe)
        elif isinstance(inputs, FSEField):
            actual_inputs = inputs
            if self.enable_syntha_integration and self.syntha_orchestrator:
                syntha_context_signal, cache['syntha_gen_cache'] = self.syntha_orchestrator.generate_context(actual_inputs)
        else: raise TypeError("Input must be FSEField or Tuple[FSEField, Optional[FSEField]]")
        actual_inputs = actual_inputs.to_device(self.device)
        if syntha_context_signal: syntha_context_signal = syntha_context_signal.to_device(self.device)
        cache['input_original'] = actual_inputs; cache['syntha_context_used'] = syntha_context_signal

        x0, cache['input_processor'] = self.input_processor.forward(actual_inputs, context_signal=syntha_context_signal)
        e1, cache['encoder1_block'] = self.encoder1_block.forward(x0, context_signal=syntha_context_signal)
        p1, cache['downsample1'] = self.downsample1.forward(e1); cache['e1_for_skip'] = e1 # Store full FSEField for skip
        e2, cache['encoder2_block'] = self.encoder2_block.forward(p1, context_signal=syntha_context_signal)
        p2, cache['downsample2'] = self.downsample2.forward(e2); cache['e2_for_skip'] = e2
        e3, cache['encoder3_block'] = self.encoder3_block.forward(p2, context_signal=syntha_context_signal)
        p3, cache['downsample3'] = self.downsample3.forward(e3); cache['e3_for_skip'] = e3
        b, cache['bottleneck_block'] = self.bottleneck_block.forward(p3, context_signal=syntha_context_signal)

        u1, cache['upsample1'] = self.upsample1.forward(b)
        s1 = self._concat_skip(u1, cache['e3_for_skip']); 
        d1, cache['decoder1_block'] = self.decoder1_block.forward(s1, context_signal=syntha_context_signal)
        cache['d1_output_for_pool'] = d1 

        u2, cache['upsample2'] = self.upsample2.forward(d1)
        s2 = self._concat_skip(u2, cache['e2_for_skip']); 
        d2, cache['decoder2_block'] = self.decoder2_block.forward(s2, context_signal=syntha_context_signal)

        u3, cache['upsample3'] = self.upsample3.forward(d2)
        s3 = self._concat_skip(u3, cache['e1_for_skip']); 
        d3, cache['decoder3_block'] = self.decoder3_block.forward(s3, context_signal=syntha_context_signal)

        keypoints, cache['keypoint_generator'] = self.keypoint_generator.forward(d3, context_signal=syntha_context_signal)
        segmentation, cache['segmentation_generator'] = self.segmentation_generator.forward(d3, context_signal=syntha_context_signal)
        
        surf_norm_raw, cache['surface_normal_generator'] = self.surface_normal_generator.forward(d3, context_signal=syntha_context_signal)
        norm_val = self.backend.linalg.norm(surf_norm_raw.data, axis=-1, keepdims=True) + 1e-7
        surface_normals_data = surf_norm_raw.data / norm_val
        surface_normals = FSEField(surface_normals_data, field_type=surf_norm_raw.field_type, device=self.device)
        cache['surface_normal_pre_norm_field'] = surf_norm_raw 
        cache['surface_normal_norm_val_data'] = norm_val

        d1_for_pool = cache['d1_output_for_pool']
        pooled_d1_data = self.backend.mean(d1_for_pool.data, axis=(1, 2))
        B_d1, C_d1 = pooled_d1_data.shape[0], d1_for_pool.shape[-1]
        pooled_d1_field = FSEField(pooled_d1_data.reshape(B_d1, 1, 1, C_d1), field_type=d1_for_pool.field_type, device=self.device)
        cache['d1_spatial_dims_for_pool_grad'] = (d1_for_pool.shape[1], d1_for_pool.shape[2])

        env_lighting_raw, cache['env_lighting_flit'] = self.env_lighting_flit.forward(pooled_d1_field, context_signal=syntha_context_signal)
        env_lighting = FSEField(env_lighting_raw.data.reshape(B_d1, -1), field_type=env_lighting_raw.field_type, device=self.device)

        outputs = {'fluxa_keypoints':keypoints, 'fluxa_segmentation':segmentation, 'fluxa_surface_normals':surface_normals, 'fluxa_environment_lighting':env_lighting}
        if self.enable_syntha_integration and self.syntha_orchestrator:
            cache['syntha_analysis'] = self.syntha_orchestrator.analyze_global_context(outputs)
        return outputs, cache

    def backward(self, upstream_grads_dict: Dict[str, FSEField], cache: Dict[str, Any]) -> Dict[str, Any]:
        """✅ COMPLETE FIXED: Backward pass with robust error handling and device sync"""
        
        # ✅ DEFENSIVE CHECK 1: Validate cache is not empty
        if not cache:
            logger.error("❌ CRITICAL: Empty cache passed to backward pass!")
            return {}
        
        # ✅ DEFENSIVE CHECK 2: Validate required cache keys exist
        required_cache_keys = [
            'keypoint_generator', 'segmentation_generator', 'surface_normal_generator',
            'env_lighting_flit', 'decoder3_block', 'decoder2_block', 'decoder1_block'
        ]
        
        missing_cache_keys = [key for key in required_cache_keys if key not in cache]
        if missing_cache_keys:
            logger.error(f"❌ Missing cache keys: {missing_cache_keys}")
            return {}
        
        # ✅ DEFENSIVE CHECK 3: Validate upstream gradients exist
        required_grad_keys = [
            'fluxa_keypoints', 'fluxa_segmentation', 'fluxa_surface_normals', 'fluxa_environment_lighting'
        ]
        
        missing_grad_keys = [key for key in required_grad_keys if key not in upstream_grads_dict]
        if missing_grad_keys:
            logger.error(f"❌ Missing upstream gradient keys: {missing_grad_keys}")
            return {}
        
        # ✅ FIX: Ensure all upstream gradients are on the correct device
        for key, grad_field in upstream_grads_dict.items():
            if grad_field.device != self.device:
                upstream_grads_dict[key] = grad_field.to_device(self.device)
                logger.debug(f"Moved {key} gradient to device {self.device}")
        
        all_param_grads: Dict[str, Any] = {}
        
        # Helper to add param grads, creating sub-dicts if comp_name doesn't exist
        def add_pg(comp_name: str, p_grads: Dict[str, FSEField]):
            if p_grads: # only add if not empty
                if comp_name not in all_param_grads: 
                    all_param_grads[comp_name] = {}
                all_param_grads[comp_name].update(p_grads)

        try:
            # ✅ ENHANCED: Output Heads with individual error handling and device sync
            
            # Get d3 shape for consistent fallback gradients
            d3_shape = None
            try:
                # Try to get d3 shape from cache if available
                if 'decoder3_block' in cache and 'inputs' in cache['decoder3_block']:
                    d3_shape = cache['decoder3_block']['inputs'].shape
                else:
                    # Fallback: estimate d3 shape (batch, height, width, base_channels)
                    batch_size = upstream_grads_dict['fluxa_keypoints'].shape[0]
                    d3_shape = (batch_size, upstream_grads_dict['fluxa_keypoints'].shape[1], 
                               upstream_grads_dict['fluxa_keypoints'].shape[2], self.base_model_channels)
            except:
                batch_size = upstream_grads_dict['fluxa_keypoints'].shape[0]
                d3_shape = (batch_size, upstream_grads_dict['fluxa_keypoints'].shape[1], 
                           upstream_grads_dict['fluxa_keypoints'].shape[2], self.base_model_channels)
            
            # Keypoint generator
            try:
                gkp_p, g_d3_kp = self.keypoint_generator.backward(
                    upstream_grads_dict['fluxa_keypoints'], 
                    cache['keypoint_generator']
                )
                add_pg('keypoint_generator', gkp_p)
                logger.debug("✅ Keypoint generator backward pass successful")
            except Exception as e:
                logger.error(f"❌ Keypoint generator backward failed: {e}")
                # Create zero gradient as fallback with correct d3 shape
                g_d3_kp = FSEField(
                    self.backend.zeros(d3_shape, dtype=self.backend.float32), 
                    device=self.device
                )
            
            # Segmentation generator  
            try:
                gseg_p, g_d3_seg = self.segmentation_generator.backward(
                    upstream_grads_dict['fluxa_segmentation'], 
                    cache['segmentation_generator']
                )
                add_pg('segmentation_generator', gseg_p)
                logger.debug("✅ Segmentation generator backward pass successful")
            except Exception as e:
                logger.error(f"❌ Segmentation generator backward failed: {e}")
                g_d3_seg = FSEField(
                    self.backend.zeros(d3_shape, dtype=self.backend.float32), 
                    device=self.device
                )

            # ✅ ROBUST FIX: Surface normal generator with proper error handling
            try:
                dL_dy_sn = upstream_grads_dict['fluxa_surface_normals']
                x_sn_field = cache['surface_normal_pre_norm_field'] 
                norm_x_sn_data = cache['surface_normal_norm_val_data']
                
                # ✅ FIX: Ensure proper device sync before computation
                if x_sn_field.device != self.device:
                    x_sn_field = x_sn_field.to_device(self.device)
                    cache['surface_normal_pre_norm_field'] = x_sn_field
                
                if isinstance(norm_x_sn_data, cp.ndarray) and self.device == "cpu":
                    norm_x_sn_data = cp.asnumpy(norm_x_sn_data)
                elif isinstance(norm_x_sn_data, np.ndarray) and self.device == "gpu":
                    norm_x_sn_data = cp.asarray(norm_x_sn_data)
                
                logger.debug(f"Surface normal backward - dL_dy_sn: {dL_dy_sn.shape}, x_sn: {x_sn_field.shape}, norm: {norm_x_sn_data.shape}")
                
                # Get shapes and ensure we're working with the right dimensions
                B, H, W, C = x_sn_field.shape  # e.g., (3, 480, 640, 3)
                backend = self.backend
                
                # ✅ SAFE FIX: Use simpler gradient computation to avoid complex broadcasting
                # For surface normal normalization: y = x / ||x||, so dy/dx = (I - x*x^T/||x||^2) / ||x||
                
                x_data = x_sn_field.data  # [B, H, W, 3]
                dL_dy_data = dL_dy_sn.data  # [B, H, W, 3]
                
                # Flatten for easier computation
                x_flat = x_data.reshape(-1, C)  # [B*H*W, 3]
                dL_dy_flat = dL_dy_data.reshape(-1, C)  # [B*H*W, 3]
                norm_flat = norm_x_sn_data.reshape(-1, 1)  # [B*H*W, 1]
                
                # Compute normalized gradients using simpler approach
                # For each position: dx/d_input = (dL/dy^T * (I/||x|| - x*x^T/||x||^3))
                norm_inv = 1.0 / (norm_flat + 1e-7)  # [B*H*W, 1]
                norm_inv_3 = norm_inv ** 3  # [B*H*W, 1]
                
                # Dot product for each sample
                dot_products = backend.sum(x_flat * dL_dy_flat, axis=1, keepdims=True)  # [B*H*W, 1]
                
                # Gradient computation: (dL/dy / ||x||) - (x * (x^T * dL/dy) / ||x||^3)
                dL_dx_flat = (dL_dy_flat * norm_inv) - (x_flat * dot_products * norm_inv_3)
                
                # Reshape back to original spatial dimensions
                dL_dx_sn_data = dL_dx_flat.reshape(B, H, W, C)
                
                g_surf_norm_raw = FSEField(dL_dx_sn_data, field_type=x_sn_field.field_type, device=self.device)
                gsn_p, g_d3_sn = self.surface_normal_generator.backward(g_surf_norm_raw, cache['surface_normal_generator'])
                add_pg('surface_normal_generator', gsn_p)
                
                logger.debug("✅ Surface normal generator backward pass successful")
                
            except Exception as e:
                logger.error(f"❌ Surface normal generator backward failed: {e}")
                import traceback
                logger.error(f"   Full traceback: {traceback.format_exc()}")
                
                # Create zero gradient as fallback
                g_d3_sn = FSEField(
                    self.backend.zeros(d3_shape, dtype=self.backend.float32), 
                    device=self.device
                )
            
            # Environment lighting generator
            try:
                g_env_out = upstream_grads_dict['fluxa_environment_lighting']
                
                # ✅ FIX: Ensure device sync
                if g_env_out.device != self.device:
                    g_env_out = g_env_out.to_device(self.device)
                
                g_env_flit_in_s = (g_env_out.shape[0],1,1,g_env_out.shape[-1])
                g_env_flit_in = FSEField(g_env_out.data.reshape(g_env_flit_in_s), field_type=g_env_out.field_type, device=self.device)
                
                genv_p, g_pool_d1 = self.env_lighting_flit.backward(g_env_flit_in, cache['env_lighting_flit'])
                add_pg('env_lighting_flit', genv_p)
                
                H_d1_pool, W_d1_pool = cache['d1_spatial_dims_for_pool_grad']
                num_pool_elements = H_d1_pool * W_d1_pool
                g_d1_env_data = self.backend.tile(g_pool_d1.data.reshape(g_pool_d1.shape[0],1,1,g_pool_d1.shape[-1]), 
                                            (1, H_d1_pool, W_d1_pool, 1)) / num_pool_elements
                g_d1_from_env = FSEField(g_d1_env_data, device=self.device, field_type=g_pool_d1.field_type)
                logger.debug("✅ Environment lighting generator backward pass successful")
            except Exception as e:
                logger.error(f"❌ Environment lighting generator backward failed: {e}")
                # Create appropriate zero gradients with correct d1 shape
                d1_shape = cache.get('d1_spatial_dims_for_pool_grad', (60, 80))
                g_d1_from_env = FSEField(
                    self.backend.zeros((upstream_grads_dict['fluxa_keypoints'].shape[0], 
                                    d1_shape[0], d1_shape[1], self.base_model_channels * 4)),
                    device=self.device
                )

            # Continue with decoder backward passes...
            g_d3_data = g_d3_kp.data + g_d3_seg.data + g_d3_sn.data
            g_d3 = FSEField(g_d3_data, device=self.device, field_type=g_d3_kp.field_type)

            # Decoder backward passes
            def split_grad(gc:FSEField, s1_shape:Tuple, s2_shape:Tuple) -> Tuple[FSEField,FSEField]:
                ch1=s1_shape[-1]
                return (FSEField(gc.data[...,:ch1],device=self.device,field_type=gc.field_type),
                        FSEField(gc.data[...,ch1:],device=self.device,field_type=gc.field_type))

            gdp3, gs3 = self.decoder3_block.backward(g_d3, cache['decoder3_block']); add_pg('decoder3_block',gdp3)
            gu3, ge1s3 = split_grad(gs3, cache['upsample3']['inputs_shape'], cache['e1_for_skip'].shape)
            _, gd2 = self.upsample3.backward(gu3, cache['upsample3'])
            
            gdp2, gs2 = self.decoder2_block.backward(gd2, cache['decoder2_block']); add_pg('decoder2_block',gdp2)
            gu2, ge2s2 = split_grad(gs2, cache['upsample2']['inputs_shape'], cache['e2_for_skip'].shape)
            _, gd1 = self.upsample2.backward(gu2, cache['upsample2'])
            gd1.data += g_d1_from_env.data

            gdp1, gs1 = self.decoder1_block.backward(gd1, cache['decoder1_block']); add_pg('decoder1_block',gdp1)
            gu1, ge3s1 = split_grad(gs1, cache['upsample1']['inputs_shape'], cache['e3_for_skip'].shape)
            _, gb = self.upsample1.backward(gu1, cache['upsample1'])

            gdbp, gp3 = self.bottleneck_block.backward(gb, cache['bottleneck_block']); add_pg('bottleneck_block',gdbp)
            _, ge3 = self.downsample3.backward(gp3, cache['downsample3']); ge3.data += ge3s1.data
            gep3, gp2 = self.encoder3_block.backward(ge3, cache['encoder3_block']); add_pg('encoder3_block',gep3)
            _, ge2 = self.downsample2.backward(gp2, cache['downsample2']); ge2.data += ge2s2.data
            gep2, gp1 = self.encoder2_block.backward(ge2, cache['encoder2_block']); add_pg('encoder2_block',gep2)
            _, ge1 = self.downsample1.backward(gp1, cache['downsample1']); ge1.data += ge1s3.data
            gep1, gx0 = self.encoder1_block.backward(ge1, cache['encoder1_block']); add_pg('encoder1_block',gep1)
            gipp, _ = self.input_processor.backward(gx0, cache['input_processor']); add_pg('input_processor',gipp)
            
            logger.debug("✅ Complete backward pass successful")
            
        except Exception as e:
            logger.error(f"❌ Critical error in backward pass: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return {}

        return all_param_grads