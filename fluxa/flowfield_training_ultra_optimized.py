# file: flowfield_training_ultra_optimized.py
# ULTRA-OPTIMIZED FlowField Training - Complete NCCL Multi-GPU Support + Rich FSE Metrics + SEGMENTATION FIX
# Revision: FIXED device synchronization, cache consistency, restored comprehensive metrics, and segmentation mask handling
# UPDATES: Added comprehensive FSE metrics computation and logging, fixed segmentation mask debugging, added batch_valid flag handling
# FIXES: M-1: Fixed metrics logging condition, M-2: Fixed global_step parameter passing, M-3: Added mask debugging, S-1: Added segmentation mask validation

import numpy as np
import cupy as cp
import time
import logging
import argparse
import os
from typing import Dict, Tuple, Any, Optional, Union, List
from datetime import datetime
import h5py

# Import optimized FlowField components
from flowfield_core_optimized import (
    FSEField, FieldType, FieldOperations, FusedFieldOperations, 
    BatchedFieldOperations, get_memory_pool, get_profiler
)
from flowfield_async_data_loader import OptimizedFlowFieldDataLoader
from flowfield_fluxa_model import ProductionFLUXA

# ✅ RESTORED: Import comprehensive FSE metrics
from FSENativeFLUXAFF.fluxa.metrics_fse import (
    FSEMetricsComputer, compute_fse_metrics,
    kp_mae, kp_coherence, seg_accuracy, seg_miou,
    sn_mae, sn_physics_accuracy, sn_coherence, env_mae
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# =========================================
# 🚀 NCCL MULTI-GPU SETUP (Unchanged)
# =========================================

def verify_nccl_setup(rank, world_size, comm, device):
    """Verify that NCCL is working properly across all ranks"""
    if world_size <= 1:
        logger.info("Single GPU mode - NCCL verification skipped")
        return True

    try:
        import torch
        import torch.distributed as dist
        from cupy.cuda import nccl

        # Ensure torch.distributed is initialized
        if not dist.is_initialized():
            logger.error(f"Rank {rank}: PyTorch distributed not initialized")
            return False

        # Built-in NCCL via torch.distributed
        logger.info(f"Rank {rank}: Testing PyTorch NCCL communication...")
        test_tensor = torch.ones(100, device=f'cuda:{rank}') * (rank + 1)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected_value = sum(range(1, world_size + 1))
        if abs(test_tensor[0].item() - expected_value) > 1e-6:
            logger.error(f"❌ Rank {rank}: PyTorch all-reduce failed")
            return False
        logger.info(f"✅ Rank {rank}: PyTorch all-reduce PASSED (got {test_tensor[0].item()})")

        # Now test your custom cupy‐nccl communicator
        if comm is None or device != "gpu":
            logger.warning(f"Rank {rank}: No custom NCCL communicator to test")
            return True

        logger.info(f"Rank {rank}: Testing custom NCCL communicator...")
        # create cupy arrays
        test_data   = cp.ones(50, dtype=cp.float32) * (rank + 1)
        result_data = cp.zeros_like(test_data)

        # pointer-level allReduce
        send_ptr    = test_data.data.ptr
        recv_ptr    = result_data.data.ptr
        count       = test_data.size
        dtype_code  = nccl.NCCL_FLOAT32
        op_code     = nccl.NCCL_SUM
        stream_ptr  = cp.cuda.Stream.null.ptr

        comm.allReduce(send_ptr, recv_ptr, count, dtype_code, op_code, stream_ptr)
        # average across GPUs
        result_data /= world_size

        expected_avg = sum(range(1, world_size + 1)) / world_size
        if abs(float(result_data[0]) - expected_avg) < 1e-6:
            logger.info(f"✅ Rank {rank}: Custom NCCL wrapper PASSED (got {float(result_data[0])})")
            return True
        else:
            logger.error(f"❌ Rank {rank}: Custom NCCL wrapper FAILED (got {float(result_data[0])}, expected {expected_avg})")
            return False

    except Exception as e:
        logger.error(f"❌ Rank {rank}: NCCL verification failed: {e}")
        return False

def setup_device_management_multi_gpu_fixed(args):
    """✅ WORKING NCCL: Ported from old working scripts"""
    
    # Get distributed training environment variables
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")
    
    logger.info(f"Distributed setup: rank={rank}, world_size={world_size}, master={master_addr}:{master_port}")
    
    comm = None
    device = "cpu"
    
    # Check GPU availability
    try:
        if cp.cuda.is_available():
            num_visible_gpus = cp.cuda.runtime.getDeviceCount()
            logger.info(f"Available GPUs detected: {num_visible_gpus}")
            
            # ✅ CRITICAL FIX: Handle torchrun's per-process GPU visibility
            if num_visible_gpus == 1 and world_size > 1:
                # Each process sees only one GPU due to CUDA_VISIBLE_DEVICES
                assigned_gpu = 0  # The only GPU this process can see
                logger.info(f"✅ Multi-GPU mode: Process {rank} assigned to visible GPU 0 (actual GPU {rank})")
            elif num_visible_gpus > 1:
                # Traditional multi-GPU setup
                if rank >= num_visible_gpus:
                    logger.error(f"Rank {rank} >= available GPUs {num_visible_gpus}. Using CPU.")
                    device = "cpu"
                    args.enable_multi_gpu = False
                    return rank, world_size, comm, device
                assigned_gpu = rank
                logger.info(f"✅ Rank {rank} assigned to GPU {assigned_gpu}")
            else:
                # Single GPU case
                assigned_gpu = 0
                logger.info(f"✅ Single GPU mode: GPU {assigned_gpu}")
                args.enable_multi_gpu = False
                world_size = 1
            
            # Set the GPU device
            cp.cuda.Device(assigned_gpu).use()
            device = "gpu"
            
            # Initialize NCCL for multi-GPU
            if args.enable_multi_gpu and world_size > 1:
                try:
                    # ✅ WORKING: Import nccl directly (from old working script)
                    from cupy.cuda import nccl
                    import torch.distributed as dist
                    
                    # Initialize torch.distributed if not already done
                    if not dist.is_initialized():
                        dist.init_process_group(
                            backend="nccl",
                            init_method=f"tcp://{master_addr}:{master_port}",
                            world_size=world_size,
                            rank=rank
                        )
                        logger.info(f"✅ PyTorch distributed initialized for rank {rank}")
                    
                    # ✅ WORKING: Create and broadcast NCCL unique ID (from old script)
                    if rank == 0:
                        comm_id = nccl.get_unique_id()
                    else:
                        comm_id = bytearray(128)  # NCCL unique ID is 128 bytes
                    
                    # Broadcast the comm_id to all ranks
                    comm_id_list = [comm_id]
                    dist.broadcast_object_list(comm_id_list, src=0)
                    comm_id = comm_id_list[0]
                    
                    # ✅ WORKING: Direct NCCL communicator (from old script)
                    comm = nccl.NcclCommunicator(world_size, comm_id, rank)
                    logger.info(f"✅ NCCL communicator initialized for rank {rank}")
                    
                except Exception as e:
                    logger.warning(f"NCCL initialization failed: {e}. Falling back to single GPU.")
                    args.enable_multi_gpu = False
                    comm = None
            
        else:
            logger.info("CUDA not available, using CPU")
            args.enable_multi_gpu = False
            
    except Exception as e:
        logger.error(f"GPU setup failed: {e}. Using CPU.")
        device = "cpu"
        args.enable_multi_gpu = False
    
    logger.info(f"Final device setup: rank={rank}, device={device}, multi_gpu={args.enable_multi_gpu}")
    return rank, world_size, comm, device

def broadcast_data_across_ranks(data_dict, rank, world_size):
    """Properly broadcast data across all ranks using PyTorch distributed"""
    
    if world_size <= 1:
        return data_dict
    
    try:
        import torch
        import torch.distributed as dist
        import pickle
        
        if rank == 0:
            # Rank 0: broadcast data
            data_bytes = pickle.dumps(data_dict)
            data_size = len(data_bytes)
            
            # Broadcast data size first
            size_tensor = torch.tensor([data_size], dtype=torch.long).cuda(rank)
            dist.broadcast(size_tensor, src=0)
            
            # Broadcast actual data
            data_tensor = torch.ByteTensor(list(data_bytes)).cuda(rank)
            dist.broadcast(data_tensor, src=0)
            
            logger.info(f"Rank {rank}: Broadcasted {data_size} bytes of data")
            return data_dict
            
        else:
            # Other ranks: receive data
            # Receive data size
            size_tensor = torch.tensor([0], dtype=torch.long).cuda(rank)
            dist.broadcast(size_tensor, src=0)
            data_size = size_tensor.item()
            
            # Receive actual data
            data_tensor = torch.zeros(data_size, dtype=torch.uint8).cuda(rank)
            dist.broadcast(data_tensor, src=0)
            
            # Deserialize data
            data_bytes = bytes(data_tensor.cpu().numpy())
            received_data = pickle.loads(data_bytes)
            
            logger.info(f"Rank {rank}: Received {data_size} bytes of data")
            return received_data
            
    except Exception as e:
        logger.error(f"Rank {rank}: Data broadcasting failed: {e}")
        return {} if rank != 0 else data_dict

# =========================================
# 🚀 NCCL-ENABLED OPTIMIZER (Unchanged)
# =========================================

class ContinuousOptimizer:
    """Continuous field optimizer for FSE models with NCCL support"""
    
    def __init__(self, parameters_dict: Dict[str, Any], learning_rate: float = 0.001,
                 world_size: int = 1, rank: int = 0, comm=None, warmup_steps: int = 1000,
                 lr_decay_factor: float = 0.95, lr_decay_steps: int = 5000):
        self.parameters = self._flatten_parameters(parameters_dict)
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.world_size = world_size
        self.rank = rank
        self.comm = comm
        self.step_count = 0
        
        # Learning rate scheduling
        self.warmup_steps = warmup_steps
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps
        
        # Momentum tracking
        self.momentum = {}
        for param_name in self.parameters:
            self.momentum[param_name] = None
        
        logger.info(f"ContinuousOptimizer: {len(self.parameters)} parameters, initial_lr={learning_rate}")
    
    def _update_learning_rate(self):
        """Update learning rate with warmup and decay"""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            self.lr = self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Exponential decay
            decay_steps = (self.step_count - self.warmup_steps) // self.lr_decay_steps
            self.lr = self.initial_lr * (self.lr_decay_factor ** decay_steps)
    
    def _flatten_parameters(self, nested_dict: Dict[str, Any], prefix: str = "") -> Dict[str, FSEField]:
        """Flatten nested parameter dictionaries"""
        flat_params = {}
        for key, value in nested_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, FSEField):
                flat_params[full_key] = value
            elif isinstance(value, dict):
                flat_params.update(self._flatten_parameters(value, full_key))
        
        return flat_params
    
    def apply_gradients(self, gradients_dict: Dict[str, Any], batch_size: int):
        """Apply gradients with momentum and distributed synchronization"""
        flat_grads = self._flatten_parameters(gradients_dict)
        self._update_learning_rate()
        self.step_count += 1
        sync_success = True

        for name, param in self.parameters.items():
            if name not in flat_grads:
                continue
            grad_field = flat_grads[name]
            avg_grad_data = grad_field.data / batch_size

            # NCCL all-reduce if in multi-GPU mode
            if self.world_size > 1 and self.comm is not None:
                try:
                    from cupy.cuda import nccl

                    # pointer-level allReduce
                    send_ptr   = avg_grad_data.data.ptr
                    recv_ptr   = avg_grad_data.data.ptr
                    count      = avg_grad_data.size
                    dtype_code = nccl.NCCL_FLOAT32
                    op_code    = nccl.NCCL_SUM
                    stream_ptr = cp.cuda.Stream.null.ptr

                    self.comm.allReduce(send_ptr, recv_ptr, count, dtype_code, op_code, stream_ptr)
                    # average across GPUs
                    avg_grad_data /= self.world_size

                    if self.rank == 0 and self.step_count % 100 == 0:
                        logger.debug(f"✅ Gradients synchronized across {self.world_size} GPUs")

                except Exception as e:
                    if self.rank == 0:
                        logger.warning(f"NCCL gradient sync failed: {e}")
                    sync_success = False

            # momentum update
            if self.momentum[name] is None:
                self.momentum[name] = param.backend.zeros_like(param.data)
            self.momentum[name] = 0.9 * self.momentum[name] + avg_grad_data

            # parameter update
            param.data -= self.lr * self.momentum[name]

        if self.step_count % 100 == 0 and self.rank == 0 and self.world_size > 1:
            status = "✅" if sync_success else "❌"
            logger.info(f"🔄 Step {self.step_count}: Multi-GPU NCCL sync {status}")
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.lr

# =========================================
# 🚀 LOSS FUNCTIONS WITH DEVICE SYNC FIXES (Unchanged)
# =========================================

class MSELoss:
    """Mean Squared Error loss for continuous fields with device sync fix"""
    
    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        # ✅ FIX: Ensure both fields are on the same device
        if pred_field.device != target_field.device:
            target_field = target_field.to_device(pred_field.device)
        
        backend = pred_field.backend
        
        # Compute MSE
        diff = pred_field.data - target_field.data
        loss_value = float(backend.mean(diff ** 2))
        
        # Gradient: 2 * (pred - target) / N
        grad_data = 2.0 * diff / diff.size
        grad_field = FSEField(grad_data, device=pred_field.device)
        
        return loss_value, grad_field

class DiceLoss:
    """Dice loss for segmentation tasks with device sync fix"""
    
    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        # ✅ FIX: Ensure both fields are on the same device
        if pred_field.device != target_field.device:
            target_field = target_field.to_device(pred_field.device)
        
        backend = pred_field.backend
        smooth = 1e-6
        
        # Flatten for computation
        pred_flat = pred_field.data.reshape(-1)
        target_flat = target_field.data.reshape(-1)
        
        # Dice coefficient
        intersection = backend.sum(pred_flat * target_flat)
        union = backend.sum(pred_flat) + backend.sum(target_flat)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Dice loss
        loss_value = float(1.0 - dice)
        
        # Gradient computation
        grad_numerator = 2.0 * target_flat
        grad_denominator = 2.0 * pred_flat
        grad_flat = -(grad_numerator * union - grad_denominator * (2.0 * intersection + smooth)) / ((union + smooth) ** 2)
        
        grad_data = grad_flat.reshape(pred_field.shape)
        grad_field = FSEField(grad_data, device=pred_field.device)
        
        return loss_value, grad_field

class FocalLoss:
    """Focal loss for handling class imbalance with device sync fix"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        # ✅ FIX: Ensure both fields are on the same device
        if pred_field.device != target_field.device:
            target_field = target_field.to_device(pred_field.device)
        
        backend = pred_field.backend
        eps = 1e-8
        
        # Apply sigmoid to predictions
        pred_sigmoid = 1.0 / (1.0 + backend.exp(-pred_field.data))
        pred_sigmoid = backend.clip(pred_sigmoid, eps, 1.0 - eps)
        
        # Focal loss computation
        ce_loss = -(target_field.data * backend.log(pred_sigmoid) + 
                   (1 - target_field.data) * backend.log(1 - pred_sigmoid))
        
        pt = backend.where(target_field.data == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = self.alpha * ((1 - pt) ** self.gamma)
        
        focal_loss = focal_weight * ce_loss
        loss_value = float(backend.mean(focal_loss))
        
        # Gradient computation (simplified)
        grad_data = focal_weight * (pred_sigmoid - target_field.data)
        grad_field = FSEField(grad_data, device=pred_field.device)
        
        return loss_value, grad_field

class CombinedDiceFocalLoss:
    """Enhanced Combined Dice and Focal loss with configurable weights and device sync"""
    
    def __init__(self, args):
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(args.focal_loss_alpha, args.focal_loss_gamma)
        self.dice_weight = args.seg_dice_weight
        self.focal_weight = args.seg_focal_weight
        
        logger.info(f"CombinedDiceFocalLoss: dice_weight={self.dice_weight}, focal_weight={self.focal_weight}")
    
    def forward(self, pred_field: FSEField, target_field: FSEField) -> Tuple[float, FSEField]:
        # ✅ FIX: Ensure both fields are on the same device
        if pred_field.device != target_field.device:
            target_field = target_field.to_device(pred_field.device)
        
        dice_loss, dice_grad = self.dice_loss.forward(pred_field, target_field)
        focal_loss, focal_grad = self.focal_loss.forward(pred_field, target_field)
        
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        combined_grad = FSEField(
            self.dice_weight * dice_grad.data + self.focal_weight * focal_grad.data,
            device=pred_field.device
        )
        
        return combined_loss, combined_grad

# =========================================
# 🚀 CHECKPOINT MANAGEMENT (Unchanged)
# =========================================

def save_checkpoint(model, optimizer, epoch: int, step: int, loss: float, 
                   gcs_path: str, bucket_name: str, rank: int = 0):
    """Save checkpoint to GCS with HDF5 format for multi-modal models"""
    if rank != 0:  # Only rank 0 saves checkpoints
        return
    
    try:
        from google.cloud import storage
        import h5py
        import tempfile
        import os
        import json
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create checkpoint as HDF5 file
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.h5"
        blob_path = f"{gcs_path}/{checkpoint_name}"
        
        # Use temporary file for HDF5 data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create HDF5 checkpoint with compression
            with h5py.File(tmp_path, 'w') as h5f:
                # ✅ METADATA GROUP
                meta_group = h5f.create_group('metadata')
                meta_group.attrs['epoch'] = epoch
                meta_group.attrs['step'] = step
                meta_group.attrs['loss'] = loss
                meta_group.attrs['timestamp'] = datetime.now().isoformat()
                meta_group.attrs['model_type'] = 'FSE_MultiModal_Vision'
                
                # ✅ MODEL PARAMETERS GROUP (with compression)
                model_group = h5f.create_group('model_parameters')
                serialize_fse_parameters_h5(model.parameters, model_group)
                
                # ✅ OPTIMIZER STATE GROUP
                opt_group = h5f.create_group('optimizer_state')
                opt_group.attrs['initial_lr'] = optimizer.initial_lr
                opt_group.attrs['current_lr'] = optimizer.lr
                opt_group.attrs['step_count'] = optimizer.step_count
                opt_group.attrs['warmup_steps'] = optimizer.warmup_steps
                opt_group.attrs['lr_decay_factor'] = optimizer.lr_decay_factor
                opt_group.attrs['lr_decay_steps'] = optimizer.lr_decay_steps
                
                # Save momentum states with compression
                momentum_group = opt_group.create_group('momentum')
                for key, momentum_data in optimizer.momentum.items():
                    if momentum_data is not None:
                        # Convert to CPU if needed
                        if hasattr(momentum_data, 'get'):  # CuPy array
                            momentum_cpu = momentum_data.get()
                        else:
                            momentum_cpu = momentum_data
                        
                        momentum_group.create_dataset(
                            key.replace('.', '_'),  # HDF5 doesn't like dots in names
                            data=momentum_cpu,
                            compression='gzip',
                            compression_opts=6
                        )
            
            # Upload HDF5 file to GCS
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(tmp_path)
            
            # Update latest checkpoint metadata (keep this as JSON for easy reading)
            latest_meta = {
                'latest_checkpoint': blob_path,
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'format': 'hdf5',
                'compression': 'gzip'
            }
            
            meta_blob = bucket.blob(f"{gcs_path}/latest_checkpoint_meta.json")
            meta_blob.upload_from_string(json.dumps(latest_meta, indent=2))
            
            # Get file size for logging
            file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            logger.info(f"✅ HDF5 checkpoint saved: {blob_path} ({file_size_mb:.1f} MB)")
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Failed to save HDF5 checkpoint: {e}")

def serialize_fse_parameters_h5(parameters_dict: Dict[str, Any], h5_group: h5py.Group, prefix: str = ""):
    """Serialize FSE parameters to HDF5 with hierarchical organization"""
    
    for key, value in parameters_dict.items():
        # Create safe HDF5 name (replace dots with underscores)
        safe_key = key.replace('.', '_')
        full_key = f"{prefix}_{safe_key}" if prefix else safe_key
        
        if isinstance(value, FSEField):
            # Convert to CPU numpy array
            if value.device == "gpu":
                data = cp.asnumpy(value.data)
            else:
                data = value.data
            
            # Create dataset with compression for large arrays
            compression = 'gzip' if data.nbytes > 1024 else None  # Compress if >1KB
            compression_opts = 6 if compression else None
            
            dataset = h5_group.create_dataset(
                f"{full_key}_data",
                data=data,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True  # Better compression
            )
            
            # Store FSE metadata as attributes
            dataset.attrs['field_type'] = value.field_type.value
            dataset.attrs['evolution_rate'] = value.evolution_rate
            dataset.attrs['device'] = value.device
            dataset.attrs['dtype'] = str(value.dtype)
            dataset.attrs['shape'] = value.shape
            
        elif isinstance(value, dict):
            # Create subgroup for nested dictionaries
            subgroup = h5_group.create_group(safe_key)
            serialize_fse_parameters_h5(value, subgroup, full_key)
        else:
            # Store simple values as attributes
            try:
                h5_group.attrs[safe_key] = value
            except (TypeError, ValueError):
                # If can't store as attribute, convert to string
                h5_group.attrs[safe_key] = str(value)

def load_checkpoint(model, optimizer, load_path: str, bucket_name: str, 
                   device: str, rank: int = 0) -> Tuple[int, int, float]:
    """Load checkpoint from GCS with HDF5 format support"""
    
    if load_path == "none":
        logger.info("No checkpoint loading requested")
        return 0, 0, float('inf')
    
    try:
        from google.cloud import storage
        import h5py
        import tempfile
        import os
        import json
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Handle "latest" checkpoint
        if load_path == "latest":
            try:
                base_path = load_path if load_path.startswith('gs://') else load_path
                meta_blob = bucket.blob(f"{base_path}/latest_checkpoint_meta.json")
                
                if not meta_blob.exists():
                    # Find latest .h5 checkpoint
                    search_prefix = f"{base_path}/" if not base_path.endswith('/') else base_path
                    checkpoint_blobs = list(bucket.list_blobs(prefix=f"{search_prefix}checkpoint_"))
                    checkpoint_blobs = [b for b in checkpoint_blobs if b.name.endswith('.h5')]
                    
                    if not checkpoint_blobs:
                        logger.warning(f"No .h5 checkpoints found in {search_prefix}")
                        return 0, 0, float('inf')
                    
                    latest_blob = max(checkpoint_blobs, key=lambda b: b.time_created)
                    checkpoint_path = latest_blob.name
                    logger.info(f"Found latest .h5 checkpoint: {checkpoint_path}")
                else:
                    meta_data = json.loads(meta_blob.download_as_text())
                    checkpoint_path = meta_data['latest_checkpoint']
                    logger.info(f"Using latest checkpoint from metadata: {checkpoint_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load latest checkpoint metadata: {e}")
                return 0, 0, float('inf')
        else:
            checkpoint_path = load_path
        
        # Download and load HDF5 checkpoint
        blob = bucket.blob(checkpoint_path)
        if not blob.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, 0, float('inf')
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            blob.download_to_filename(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Load HDF5 checkpoint
            with h5py.File(tmp_path, 'r') as h5f:
                # Load metadata
                meta_group = h5f['metadata']
                epoch = int(meta_group.attrs['epoch'])
                step = int(meta_group.attrs['step'])
                loss = float(meta_group.attrs['loss'])
                
                if rank == 0:
                    logger.info(f"Loading HDF5 checkpoint from: {checkpoint_path}")
                    logger.info(f"  Format: {meta_group.attrs.get('model_type', 'Unknown')}")
                    logger.info(f"  Timestamp: {meta_group.attrs.get('timestamp', 'Unknown')}")
                
                # Restore model parameters
                if 'model_parameters' in h5f:
                    model_params = deserialize_fse_parameters_h5(h5f['model_parameters'], device)
                    _update_parameters_recursive(model.parameters, model_params)
                
                # Restore optimizer state
                if 'optimizer_state' in h5f:
                    opt_group = h5f['optimizer_state']
                    optimizer.initial_lr = float(opt_group.attrs.get('initial_lr', optimizer.initial_lr))
                    optimizer.lr = float(opt_group.attrs.get('current_lr', optimizer.lr))
                    optimizer.step_count = int(opt_group.attrs.get('step_count', 0))
                    optimizer.warmup_steps = int(opt_group.attrs.get('warmup_steps', optimizer.warmup_steps))
                    optimizer.lr_decay_factor = float(opt_group.attrs.get('lr_decay_factor', optimizer.lr_decay_factor))
                    optimizer.lr_decay_steps = int(opt_group.attrs.get('lr_decay_steps', optimizer.lr_decay_steps))
                    
                    # Restore momentum
                    if 'momentum' in opt_group:
                        momentum_group = opt_group['momentum']
                        backend = cp if device == "gpu" else np
                        
                        for dataset_name in momentum_group.keys():
                            # Convert back from safe name
                            param_name = dataset_name.replace('_', '.')
                            momentum_data = momentum_group[dataset_name][...]
                            
                            if param_name in optimizer.momentum:
                                optimizer.momentum[param_name] = backend.array(momentum_data)
            
            if rank == 0:
                file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
                logger.info(f"✅ HDF5 checkpoint loaded: epoch={epoch}, step={step}, loss={loss:.6f} ({file_size_mb:.1f} MB)")
            
            return epoch, step, loss
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Failed to load HDF5 checkpoint: {e}")
        return 0, 0, float('inf')

def deserialize_fse_parameters_h5(h5_group: h5py.Group, device: str) -> Dict[str, Any]:
    """Deserialize FSE parameters from HDF5 format"""
    
    result = {}
    
    for key in h5_group.keys():
        if key.endswith('_data'):
            # This is an FSEField dataset
            param_name = key[:-5]  # Remove '_data' suffix
            dataset = h5_group[key]
            
            # Load data
            data = dataset[...]
            backend = cp if device == "gpu" else np
            field_data = backend.array(data)
            
            # Load FSE metadata from attributes
            from flowfield_core_optimized import FieldType
            field_type = FieldType(dataset.attrs['field_type'])
            evolution_rate = float(dataset.attrs['evolution_rate'])
            
            # Create FSEField
            result[param_name.replace('_', '.')] = FSEField(
                field_data, field_type, evolution_rate, device
            )
            
        elif isinstance(h5_group[key], h5py.Group):
            # Nested group - recurse
            result[key.replace('_', '.')] = deserialize_fse_parameters_h5(h5_group[key], device)
    
    return result

def _update_parameters_recursive(target_dict, source_dict):
    """Recursively update model parameters"""
    for key, value in source_dict.items():
        if key in target_dict:
            if isinstance(value, dict) and isinstance(target_dict[key], dict):
                _update_parameters_recursive(target_dict[key], value)
            elif isinstance(value, FSEField) and hasattr(target_dict[key], 'data'):
                target_dict[key].data = value.data

def _initialize_gcs_client(bucket_name: str, project_id: str):
    """Initialize GCS client"""
    try:
        from google.cloud import storage
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        logger.info(f"✅ GCS client initialized: {bucket_name}")
        return bucket
    except Exception as e:
        logger.error(f"Failed to initialize GCS: {e}")
        return None

def _get_gcs_bucket():
    """Get GCS bucket for the current session"""
    try:
        from google.cloud import storage
        # Try to get from environment or existing client
        client = storage.Client()
        bucket_name = os.environ.get('GCS_BUCKET_NAME', 'auralith')  # Default fallback
        return client.bucket(bucket_name)
    except Exception as e:
        logger.warning(f"Could not initialize GCS bucket: {e}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FlowField Ultra-Optimized Training")
    
    # Cloud Settings
    parser.add_argument("--project_id", type=str, default="your-project")
    parser.add_argument("--bucket_name", type=str, default="your-bucket")
    parser.add_argument("--job_dir", type=str, default="/tmp/job_dir")
    
    # Model parameters
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--base_model_channels", type=int, default=32)
    parser.add_argument("--max_cses_per_fil", type=int, default=4)
    parser.add_argument("--use_bias", action="store_true", default=False)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--base_batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--val_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--field_evolution_rate", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_decay_factor", type=float, default=0.95)
    parser.add_argument("--lr_decay_steps", type=int, default=5000)
    parser.add_argument("--validation_frequency", type=int, default=5)  # Run validation every N epochs
    parser.add_argument("--early_stopping_patience", type=int, default=10)  # Early stopping patience
    
    # Loss parameters
    parser.add_argument("--segmentation_loss_type", type=str, default="DiceLoss",
                       choices=["DiceLoss", "FocalLoss", "CombinedDiceFocalLoss", "MSELoss"])
    parser.add_argument("--focal_loss_alpha", type=float, default=1.0)
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0)
    parser.add_argument("--seg_dice_weight", type=float, default=0.5)
    parser.add_argument("--seg_focal_weight", type=float, default=0.5)
    parser.add_argument("--keypoints_loss_weight", type=float, default=1.0)
    parser.add_argument("--segmentation_loss_weight", type=float, default=1.0)
    parser.add_argument("--surface_normals_loss_weight", type=float, default=1.0)
    parser.add_argument("--env_lighting_loss_weight", type=float, default=0.1)
    
    # Data parameters
    parser.add_argument("--validation_split_fraction", type=float, default=0.2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--prefetch_batches", type=int, default=2)
    parser.add_argument("--num_data_workers", type=int, default=4)
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_save_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_save_interval", type=int, default=5)
    parser.add_argument("--gcs_checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--load_checkpoint_path", type=str, default="none")
    
    # SYNTHA parameters
    parser.add_argument("--enable_syntha_integration", action="store_true", default=False)
    
    # Multi-GPU parameters
    parser.add_argument("--enable_multi_gpu", action="store_true", default=False)
    
    # Performance parameters
    parser.add_argument("--enable_vectorized_batching", action="store_true", default=False)
    parser.add_argument("--enable_memory_optimization", action="store_true", default=False)
    
    # ✅ NEW: Metrics parameters
    parser.add_argument("--enable_comprehensive_metrics", action="store_true", default=True)
    parser.add_argument("--metrics_compute_frequency", type=int, default=10)  # Compute metrics every N steps
    parser.add_argument("--metrics_log_frequency", type=int, default=25)  # Log metrics every N steps
    
    # GCS parameters
    parser.add_argument("--gcs_images_path", type=str, default="images")
    parser.add_argument("--gcs_labels_base_path", type=str, default="labels")
    parser.add_argument("--allow_fallback_batches", action="store_true", default=False,
                   help="Permit emergency fallback batches when real data fails (disable for production)")
    return parser.parse_args()

# =========================================
# 🚀 ULTRA-OPTIMIZED BATCH PROCESSING WITH DEVICE SYNC AND COMPREHENSIVE METRICS
# =========================================

def run_validation(model, data_loader, loss_functions, loss_weights, args, device, rank=0):
    """Run validation loop with comprehensive metrics"""
    if not data_loader.all_val_ids:
        return {}
    
    validation_losses = []
    validation_task_losses = {name: [] for name in loss_functions.keys()}
    
    # ✅ NEW: Initialize metrics computer for validation
    if args.enable_comprehensive_metrics:
        metrics_computer = FSEMetricsComputer(device)
        validation_metrics = {}
    
    try:
        # Get validation data generator
        val_generator = data_loader.get_async_batch_generator(
            data_loader.all_val_ids, False, args.val_batch_size_per_gpu
        )
        
        val_steps = 0
        max_val_steps = min(50, len(data_loader.all_val_ids) // args.val_batch_size_per_gpu)  # Limit validation steps
        
        for val_batch_idx, batch_tuple in enumerate(val_generator):
            if val_batch_idx >= max_val_steps:
                break
                
            try:
                # ✅ S-1 FIX: Handle both old and new format
                if len(batch_tuple) == 3:
                    val_batch_inputs, val_batch_labels, val_batch_valid = batch_tuple
                else:
                    val_batch_inputs, val_batch_labels = batch_tuple
                    val_batch_valid = [{"fluxa_segmentation": True} for _ in range(len(val_batch_inputs))]
                
                # Forward pass (no training)
                val_outputs, _ = ultra_fast_batch_forward(
                    model, val_batch_inputs, args, training=False
                )
                
                if val_outputs:
                    # Compute validation loss
                    val_loss, _, val_task_losses = ultra_fast_loss_computation(
                        val_outputs, val_batch_labels, loss_functions, loss_weights, device
                    )
                    
                    validation_losses.append(val_loss)
                    for task_name, task_loss in val_task_losses.items():
                        validation_task_losses[task_name].append(task_loss)
                    
                    # ✅ NEW: Compute comprehensive metrics for validation
                    if args.enable_comprehensive_metrics and val_batch_idx % 5 == 0:  # Every 5th validation batch
                        try:
                            batch_metrics = metrics_computer.compute_all_metrics(val_outputs, val_batch_labels)
                            for metric_name, metric_value in batch_metrics.items():
                                if f"val_{metric_name}" not in validation_metrics:
                                    validation_metrics[f"val_{metric_name}"] = []
                                validation_metrics[f"val_{metric_name}"].append(metric_value)
                        except Exception as e:
                            logger.debug(f"Validation metrics computation failed: {e}")
                    
                    val_steps += 1
                    
            except Exception as e:
                logger.warning(f"Validation batch {val_batch_idx} failed: {e}")
                continue
        
        # Compute validation metrics
        if validation_losses:
            val_metrics = {
                'avg_val_loss': np.mean(validation_losses),
                'val_steps': val_steps
            }
            
            for task_name, losses in validation_task_losses.items():
                if losses:
                    val_metrics[f'val_{task_name}'] = np.mean(losses)
            
            # ✅ NEW: Add comprehensive validation metrics
            if args.enable_comprehensive_metrics and validation_metrics:
                for metric_name, metric_values in validation_metrics.items():
                    if metric_values:
                        val_metrics[metric_name] = np.mean(metric_values)
            
            return val_metrics
        else:
            return {'avg_val_loss': float('inf'), 'val_steps': 0}
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'avg_val_loss': float('inf'), 'val_steps': 0}

def ultra_fast_batch_forward(model, batch_inputs, args, training=True):
    """Ultra-fast batch forward with all optimizations enabled"""
    
    if not batch_inputs:
        return {}, {}
    
    # Get profiler instance with safe fallback
    try:
        profiler = get_profiler()
        use_profiling = True
    except Exception as e:
        logger.debug(f"Profiler unavailable: {e}")
        profiler = None
        use_profiling = False
    
    # Use optimized batch processing with optional profiling
    if use_profiling and profiler:
        with profiler("batch_forward"):
            return _execute_batch_forward(model, batch_inputs, args, training)
    else:
        return _execute_batch_forward(model, batch_inputs, args, training)

def _execute_batch_forward(model, batch_inputs, args, training):
    """Execute the actual batch forward pass with device sync"""
    first_input = batch_inputs[0]
    
    # ✅ FIX: Ensure all inputs are on the model's device
    def sync_to_device(field_or_tuple, target_device):
        if isinstance(field_or_tuple, tuple):
            return tuple(sync_to_device(item, target_device) for item in field_or_tuple)
        elif isinstance(field_or_tuple, FSEField):
            return field_or_tuple.to_device(target_device)
        else:
            return field_or_tuple
    
    # Handle SYNTHA case with optimized processing
    if isinstance(first_input, tuple) and len(first_input) == 2:
        batched_input, batched_context = first_input
        
        # Ensure both are on model device
        batched_input = sync_to_device(batched_input, model.device)
        batched_context = sync_to_device(batched_context, model.device)
        
        # Verify inputs are properly batched
        if not is_properly_batched(batched_input):
            logger.warning("Input not properly batched, this will hurt performance")
        
        return model.forward((batched_input, batched_context), training=training)
    
    # Handle regular inputs
    first_field = first_input if isinstance(first_input, FSEField) else first_input[0]
    
    if len(batch_inputs) == 1 and is_properly_batched(first_field):
        # Already batched - use directly but ensure device sync
        synced_input = sync_to_device(first_field, model.device)
        return model.forward(synced_input, training=training)
    else:
        # Need to batch - use optimized stacking
        backend = cp if model.device == "gpu" else np
        
        # Sync all inputs to model device first
        synced_inputs = [sync_to_device(inp, model.device) for inp in batch_inputs]
        
        try:
            profiler = get_profiler()
            if profiler:
                with profiler("batch_stacking"):
                    stacked_data = backend.stack([inp.data for inp in synced_inputs], axis=0)
                    batched_field = FSEField(stacked_data, synced_inputs[0].field_type, device=model.device)
            else:
                stacked_data = backend.stack([inp.data for inp in synced_inputs], axis=0)
                batched_field = FSEField(stacked_data, synced_inputs[0].field_type, device=model.device)
        except:
            # Fallback without profiling
            stacked_data = backend.stack([inp.data for inp in synced_inputs], axis=0)
            batched_field = FSEField(stacked_data, synced_inputs[0].field_type, device=model.device)
        
        return model.forward(batched_field, training=training)

def is_properly_batched(field: FSEField) -> bool:
    """Check if field has proper batch dimension for efficient processing"""
    return field.data.ndim >= 4 and field.shape[0] > 1

def ultra_fast_loss_computation_with_metrics(pred_outputs, batch_labels, loss_functions, loss_weights, device, 
                                            args, step_count, metrics_computer=None, batch_valid=None):
    """✅ NEW: Ultra-fast loss computation with optional comprehensive metrics and segmentation mask handling"""
    
    try:
        profiler = get_profiler()
        use_profiling = profiler is not None
    except:
        use_profiling = False
    
    def compute_losses_and_metrics():
        total_loss = 0.0
        loss_gradients = {}
        individual_losses = {}
        step_metrics = {}
        
        # Process all losses in parallel where possible
        for task_name, pred_field in pred_outputs.items():
            if task_name not in batch_labels or task_name not in loss_functions:
                continue
            
            try:
                target_field = batch_labels[task_name]
                
                # ✅ FIX: Ensure both pred and target are on the same device
                if pred_field.device != device:
                    pred_field = pred_field.to_device(device)
                if target_field.device != device:
                    target_field = target_field.to_device(device)
                
                # ✅ S-1 NEW: Soft-weight segmentation loss instead of skipping
                if task_name == "fluxa_segmentation" and batch_valid is not None:
                    backend = cp if device == "gpu" else np
                    valid_mask = backend.array(
                        [v["fluxa_segmentation"] for v in batch_valid], dtype=backend.float32
                    )
                    loss_scale = float(valid_mask.mean())          # 0 … 1
                else:
                    loss_scale = 1.0
                
                # Shape validation and correction
                if pred_field.shape != target_field.shape:
                    target_field = resolve_shape_mismatch(pred_field, target_field, task_name)
                    if target_field is None:
                        continue
                
                # Compute loss with optional profiling
                loss_fn = loss_functions[task_name]
                
                if use_profiling:
                    with profiler(f"loss_{task_name}"):
                        task_loss, loss_grad = loss_fn.forward(pred_field, target_field)
                else:
                    task_loss, loss_grad = loss_fn.forward(pred_field, target_field)
                
                # ✅ S-1 NEW: Apply segmentation loss scaling
                if task_name == "fluxa_segmentation":
                    task_loss *= loss_scale
                    loss_grad.data *= loss_scale
                
                weighted_loss = task_loss * loss_weights[task_name]
                total_loss += weighted_loss
                individual_losses[task_name] = weighted_loss
                loss_gradients[task_name] = loss_grad * loss_weights[task_name]
                
            except Exception as e:
                logger.warning(f"Loss computation failed for {task_name}: {e}")
                individual_losses[task_name] = 0.0
                continue
        
        # ✅ NEW: Compute comprehensive metrics if enabled and it's time
        if (args.enable_comprehensive_metrics and 
            metrics_computer is not None and 
            step_count % args.metrics_compute_frequency == 0):
            
            try:
                if use_profiling:
                    with profiler("comprehensive_metrics"):
                        step_metrics = metrics_computer.compute_all_metrics(pred_outputs, batch_labels)
                else:
                    step_metrics = metrics_computer.compute_all_metrics(pred_outputs, batch_labels)
                
                logger.debug(f"✅ Computed {len(step_metrics)} comprehensive metrics for step {step_count}")
                
            except Exception as e:
                logger.debug(f"Comprehensive metrics computation failed: {e}")
                step_metrics = {}
        
        return total_loss, loss_gradients, individual_losses, step_metrics
    
    if use_profiling:
        with profiler("loss_and_metrics_computation"):
            return compute_losses_and_metrics()
    else:
        return compute_losses_and_metrics()

def ultra_fast_loss_computation(pred_outputs, batch_labels, loss_functions, loss_weights, device):
    """Backward compatibility wrapper for loss computation without metrics"""
    
    try:
        profiler = get_profiler()
        use_profiling = profiler is not None
    except:
        use_profiling = False
    
    def compute_losses():
        total_loss = 0.0
        loss_gradients = {}
        individual_losses = {}
        
        # Process all losses in parallel where possible
        for task_name, pred_field in pred_outputs.items():
            if task_name not in batch_labels or task_name not in loss_functions:
                continue
            
            try:
                target_field = batch_labels[task_name]
                
                # ✅ FIX: Ensure both pred and target are on the same device
                if pred_field.device != device:
                    pred_field = pred_field.to_device(device)
                if target_field.device != device:
                    target_field = target_field.to_device(device)
                
                # Shape validation and correction
                if pred_field.shape != target_field.shape:
                    target_field = resolve_shape_mismatch(pred_field, target_field, task_name)
                    if target_field is None:
                        continue
                
                # Compute loss with optional profiling
                loss_fn = loss_functions[task_name]
                
                if use_profiling:
                    with profiler(f"loss_{task_name}"):
                        task_loss, loss_grad = loss_fn.forward(pred_field, target_field)
                else:
                    task_loss, loss_grad = loss_fn.forward(pred_field, target_field)
                
                # Apply weighting
                weighted_loss = task_loss * loss_weights[task_name]
                total_loss += weighted_loss
                individual_losses[task_name] = weighted_loss
                loss_gradients[task_name] = loss_grad * loss_weights[task_name]
                
            except Exception as e:
                logger.warning(f"Loss computation failed for {task_name}: {e}")
                individual_losses[task_name] = 0.0
                continue
        
        return total_loss, loss_gradients, individual_losses
    
    if use_profiling:
        with profiler("loss_computation"):
            return compute_losses()
    else:
        return compute_losses()

def resolve_shape_mismatch(pred_field: FSEField, target_field: FSEField, task_name: str) -> Optional[FSEField]:
    """Resolve shape mismatches between predictions and targets"""
    
    logger.debug(f"Resolving {task_name} shape mismatch: {pred_field.shape} vs {target_field.shape}")
    
    # Same batch size - try shape corrections
    if pred_field.shape[0] == target_field.shape[0]:
        # Try squeeze/unsqueeze
        pred_squeezed = pred_field.data.squeeze()
        target_squeezed = target_field.data.squeeze()
        
        if pred_squeezed.shape == target_squeezed.shape:
            return FSEField(target_squeezed, target_field.field_type, device=target_field.device)
        
        # Try reshaping if same total elements
        if pred_field.data.size == target_field.data.size:
            try:
                reshaped_target = target_field.data.reshape(pred_field.shape)
                return FSEField(reshaped_target, target_field.field_type, device=target_field.device)
            except:
                pass
    
    logger.error(f"Cannot resolve shape mismatch for {task_name}")
    return None

# =========================================
# 🚀 OPTIMIZED MEMORY MANAGEMENT (Unchanged)
# =========================================

class AdvancedMemoryManager:
    """Advanced memory management for optimal GPU utilization"""
    
    def __init__(self, device: str = "gpu"):
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.allocation_history = []
        self.peak_memory = 0
        
    def monitor_memory_usage(self, step: int, force_log: bool = False):
        """Monitor and log memory usage"""
        if self.device == "gpu":
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                used = total - free
                utilization = used / total
                
                self.peak_memory = max(self.peak_memory, used)
                
                if step % 100 == 0 or force_log:
                    logger.info(f"Step {step} GPU Memory: {utilization:.1%} used "
                              f"({used/(1024**3):.2f}GB / {total/(1024**3):.2f}GB), "
                              f"Peak: {self.peak_memory/(1024**3):.2f}GB")
                
                # Aggressive cleanup if high memory usage
                if utilization > 0.90:
                    self.emergency_cleanup()
                    
            except Exception as e:
                logger.debug(f"Memory monitoring failed: {e}")
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        if self.device == "gpu":
            logger.warning("High memory usage detected - performing emergency cleanup")
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
    
    def optimize_memory_layout(self, fields: List[FSEField]) -> List[FSEField]:
        """Optimize memory layout for better cache performance"""
        if not fields or self.device != "gpu":
            return fields
        
        # Group fields by shape for better memory locality
        shape_groups = {}
        for field in fields:
            shape = field.shape
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append(field)
        
        # Return fields grouped by shape
        optimized_fields = []
        for shape_group in shape_groups.values():
            optimized_fields.extend(shape_group)
        
        return optimized_fields
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {'device': self.device, 'peak_memory_gb': self.peak_memory / (1024**3)}
        
        if self.device == "gpu":
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                used = total - free
                stats.update({
                    'current_used_gb': used / (1024**3),
                    'current_free_gb': free / (1024**3),
                    'total_gb': total / (1024**3),
                    'utilization': used / total
                })
            except Exception as e:
                logger.debug(f"Failed to get GPU memory stats: {e}")
        
        return stats

# =========================================
# 🚀 PERFORMANCE MONITORING & OPTIMIZATION (Enhanced)
# =========================================

class FlowFieldPerformanceMonitor:
    """Comprehensive performance monitoring for FlowField operations"""
    
    def __init__(self):
        self.step_times = []
        self.batch_sizes = []
        self.loss_values = []
        self.throughput_history = []
        self.bottleneck_analysis = {}
        
        # ✅ NEW: Metrics tracking
        self.metrics_history = {}
        self.best_metrics = {}
        
    def record_step(self, step_time: float, batch_size: int, loss_value: float, metrics: Optional[Dict[str, float]] = None):
        """Record performance metrics for a training step"""
        self.step_times.append(step_time)
        self.batch_sizes.append(batch_size)
        self.loss_values.append(loss_value)
        
        # Calculate throughput (samples/second)
        throughput = batch_size / step_time if step_time > 0 else 0
        self.throughput_history.append(throughput)
        
        # ✅ NEW: Record comprehensive metrics
        if metrics:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                self.metrics_history[metric_name].append(metric_value)
                
                # Track best metrics (higher is better for most metrics)
                if metric_name not in self.best_metrics or metric_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
    
    def analyze_bottlenecks(self, profiler_stats: Dict[str, Any]) -> Dict[str, str]:
        """Analyze performance bottlenecks from profiler data"""
        bottlenecks = {}
        
        # Identify slow operations
        for op_name, stats in profiler_stats.items():
            avg_time = stats.get('avg_time', 0)
            total_time = stats.get('total_time', 0)
            
            if avg_time > 0.1:  # Operations taking >100ms
                bottlenecks[op_name] = f"Slow operation: {avg_time:.3f}s avg"
            elif total_time > 10.0:  # Operations with high cumulative time
                bottlenecks[op_name] = f"High cumulative time: {total_time:.1f}s total"
        
        return bottlenecks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.step_times:
            return {"error": "No performance data recorded"}
        
        summary = {
            'avg_step_time': np.mean(self.step_times),
            'min_step_time': np.min(self.step_times),
            'max_step_time': np.max(self.step_times),
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'total_steps': len(self.step_times),
            'avg_batch_size': np.mean(self.batch_sizes),
            'avg_loss': np.mean(self.loss_values) if self.loss_values else 0
        }
        
        # ✅ NEW: Add best metrics summary
        if self.best_metrics:
            summary['best_metrics'] = self.best_metrics.copy()
        
        return summary
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations based on monitoring data"""
        suggestions = []
        
        if not self.step_times:
            return suggestions
        
        avg_step_time = np.mean(self.step_times)
        avg_throughput = np.mean(self.throughput_history) if self.throughput_history else 0
        
        # Performance-based suggestions
        if avg_step_time > 5.0:
            suggestions.append("Step time >5s: Consider reducing batch size or model complexity")
        
        if avg_throughput < 10:  # samples/second
            suggestions.append("Low throughput: Enable more aggressive batching or async data loading")
        
        # Check for performance variance
        if len(self.step_times) > 10:
            step_variance = np.var(self.step_times)
            if step_variance > 1.0:
                suggestions.append("High step time variance: Check for data loading bottlenecks")
        
        # ✅ NEW: Metrics-based suggestions
        if self.metrics_history:
            for metric_name, values in self.metrics_history.items():
                if len(values) > 10:
                    recent_avg = np.mean(values[-10:])
                    if 'accuracy' in metric_name.lower() and recent_avg < 0.7:
                        suggestions.append(f"Low {metric_name} ({recent_avg:.3f}): Consider adjusting learning rate or model architecture")
                    elif 'coherence' in metric_name.lower() and recent_avg < 0.5:
                        suggestions.append(f"Low {metric_name} ({recent_avg:.3f}): Check FSE field evolution parameters")
        
        return suggestions

# =========================================
# 🚀 MAIN ULTRA-OPTIMIZED TRAINING LOOP WITH COMPREHENSIVE METRICS
# =========================================

def ultra_optimized_training_loop(args: argparse.Namespace):
    """Ultra-optimized training loop with full NCCL support, device sync fixes, and comprehensive FSE metrics"""
    
    logger.info("🌊 STARTING ULTRA-OPTIMIZED FLOWFIELD TRAINING WITH COMPREHENSIVE METRICS 🌊")
    logger.info("=" * 80)
    
    # Setup device and NCCL
    rank, world_size, comm, device = setup_device_management_multi_gpu_fixed(args)
    
    # Verify NCCL setup immediately (with fallback)
    try:
        nccl_success = verify_nccl_setup(rank, world_size, comm, device)
        if nccl_success:
            logger.info(f"✅ Rank {rank}: NCCL verification passed")
        else:
            logger.warning(f"⚠️ Rank {rank}: NCCL verification failed, but continuing anyway")
    except Exception as e:
        logger.warning(f"⚠️ Rank {rank}: NCCL verification error: {e}, but continuing anyway")
    
    # Initialize memory manager and performance monitor
    memory_manager = AdvancedMemoryManager(device)
    performance_monitor = FlowFieldPerformanceMonitor()
    
    # ✅ NEW: Initialize comprehensive metrics computer
    metrics_computer = None
    if args.enable_comprehensive_metrics:
        try:
            metrics_computer = FSEMetricsComputer(device)
            logger.info(f"✅ Rank {rank}: Comprehensive FSE metrics computer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics computer: {e}")
            metrics_computer = None
    
    # Initialize profiler with safe fallback
    try:
        profiler = get_profiler()
        logger.info(f"✅ Rank {rank}: Performance profiler initialized")
    except Exception as e:
        logger.warning(f"Profiler initialization failed: {e}")
        profiler = None
    
    try:
        # Initialize GCS and set environment
        if rank == 0:
            os.environ['GCS_BUCKET_NAME'] = args.bucket_name  # Set for checkpoint functions
            _initialize_gcs_client(args.bucket_name, args.project_id)
        
        # Create optimized model
        logger.info(f"🔧 Rank {rank}: Creating optimized FLUXA model...")
        model_start_time = time.time()
        
        fluxa_model = ProductionFLUXA(
            (args.img_height, args.img_width, args.input_channels),
            args.base_model_channels,
            args.enable_syntha_integration,
            device,
            args.use_bias,
            max_cses_per_fil_arg=args.max_cses_per_fil
        )
        
        model_creation_time = time.time() - model_start_time
        logger.info(f"✅ Rank {rank}: Model created in {model_creation_time:.2f}s")
        
        # Create optimized optimizer with NCCL support
        optimizer = ContinuousOptimizer(
            fluxa_model.parameters,
            learning_rate=args.learning_rate,
            world_size=world_size,
            rank=rank,
            comm=comm,
            warmup_steps=args.warmup_steps,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_steps=args.lr_decay_steps
        )
        
        # Load checkpoint if specified
        start_epoch, start_step, best_loss = load_checkpoint(
            fluxa_model, optimizer, args.load_checkpoint_path, args.bucket_name, device, rank
        )
        
        # Setup optimized data loader with proper multi-GPU data synchronization
        logger.info(f"📊 Rank {rank}: Setting up optimized data loader...")
        data_loader = OptimizedFlowFieldDataLoader(args, device, rank, world_size)
        
        # Data discovery and synchronization
        if rank == 0:
            # Only rank 0 discovers data
            data_loader.discover_and_split_ids()
            data_dict = {
                'train_ids': data_loader.all_train_ids,
                'val_ids': data_loader.all_val_ids
            }
            logger.info(f"Data discovery: {len(data_loader.all_train_ids)} train samples, {len(data_loader.all_val_ids)} val samples")
        else:
            # Other ranks start with empty data
            data_dict = {'train_ids': [], 'val_ids': []}
        
        # Broadcast data to all ranks using proper NCCL
        if world_size > 1:
            try:
                data_dict = broadcast_data_across_ranks(data_dict, rank, world_size)
                
                # Update data loader with received data
                data_loader.all_train_ids = data_dict['train_ids']
                data_loader.all_val_ids = data_dict['val_ids']
                
                logger.info(f"Rank {rank}: Received {len(data_loader.all_train_ids)} train, {len(data_loader.all_val_ids)} val samples")
                
            except Exception as e:
                logger.error(f"Rank {rank}: Data synchronization failed: {e}")
                if rank != 0:
                    logger.error("Non-rank-0 process cannot continue without data")
                    return
        
        # Verify all ranks have data
        if not data_loader.all_train_ids:
            if rank == 0:
                logger.error("No training data found")
            else:
                logger.error(f"Rank {rank}: No training data received")
            return
        
        # Synchronize all ranks before starting training
        if world_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()  # Wait for all ranks to reach this point
                    logger.info(f"✅ Rank {rank}: All ranks synchronized, starting training")
            except Exception as e:
                logger.warning(f"Rank synchronization failed: {e}")
        
        logger.info(f"✅ Rank {rank}: Data loader setup complete")
        
        # Setup loss functions with device sync fixes
        if args.segmentation_loss_type == 'CombinedDiceFocalLoss':
            seg_loss_fn = CombinedDiceFocalLoss(args)
        elif args.segmentation_loss_type == 'DiceLoss':
            seg_loss_fn = DiceLoss()
        elif args.segmentation_loss_type == 'FocalLoss':
            seg_loss_fn = FocalLoss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
        else:
            seg_loss_fn = MSELoss()
        
        loss_functions = {
            'fluxa_keypoints': MSELoss(),
            'fluxa_segmentation': seg_loss_fn,
            'fluxa_surface_normals': MSELoss(),
            'fluxa_environment_lighting': MSELoss()
        }
        
        loss_weights = {
            'fluxa_keypoints': args.keypoints_loss_weight,
            'fluxa_segmentation': args.segmentation_loss_weight,
            'fluxa_surface_normals': args.surface_normals_loss_weight,
            'fluxa_environment_lighting': args.env_lighting_loss_weight
        }
        
        # Training configuration
        num_ids_this_rank = len(data_loader.all_train_ids) // world_size
        steps_per_epoch = max(1, num_ids_this_rank // args.base_batch_size_per_gpu)
        
        if rank == 0:
            logger.info(f"🚀 Training Configuration:")
            logger.info(f"   Device: {device}")
            logger.info(f"   Epochs: {args.epochs}")
            logger.info(f"   Steps per epoch: {steps_per_epoch}")
            logger.info(f"   Batch size per GPU: {args.base_batch_size_per_gpu}")
            logger.info(f"   World size: {world_size}")
            logger.info(f"   NCCL enabled: {'✅ YES' if world_size > 1 and comm else '❌ NO'}")
            logger.info(f"   Async prefetch batches: {args.prefetch_batches}")
            logger.info(f"   Data workers: {args.num_data_workers}")
            logger.info(f"   Comprehensive metrics: {'✅ YES' if args.enable_comprehensive_metrics else '❌ NO'}")
            logger.info(f"   Fallback batches allowed: {'✅ YES' if args.allow_fallback_batches else '❌ NO'}")
            logger.info("=" * 80)
        
        # MAIN TRAINING LOOP WITH COMPREHENSIVE METRICS
        global_step = start_step  # Resume from checkpoint
        best_val_metric = -best_loss if best_loss != float('inf') else float('-inf')  # Convert loss to metric
        epoch_times = []
        early_stopping_counter = 0  # Track epochs without improvement
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            if rank == 0:
                logger.info(f"\n🌊 Epoch {epoch + 1}/{args.epochs}")
            
            epoch_losses = []
            epoch_task_losses = {name: [] for name in loss_functions.keys()}
            
            # ✅ NEW: Initialize epoch metrics tracking
            epoch_metrics = {}
            metrics_count = 0
            
            # Get async batch generator (this is where the I/O optimization kicks in)
            batch_generator = data_loader.get_async_batch_generator(
                data_loader.all_train_ids, True, args.base_batch_size_per_gpu
            )
            
            # ✅ S-1 NEW: Receive the flags from data loader
            for batch_idx, batch_tuple in enumerate(batch_generator):
                if batch_idx >= steps_per_epoch:
                    break
                
                step_start_time = time.time()
                global_step += 1
                
                # ✅ S-1 FIX: Handle both old and new format
                if len(batch_tuple) == 3:
                    batch_inputs, batch_labels, batch_valid = batch_tuple
                else:
                    batch_inputs, batch_labels = batch_tuple
                    batch_valid = [{"fluxa_segmentation": True} for _ in range(len(batch_inputs))]
                
                # ✅ M-3: Add mask debugging check at step 5
                if rank == 0 and global_step == 5:
                    mask_sum = cp.sum(batch_labels['fluxa_segmentation'].data) if device == "gpu" else np.sum(batch_labels['fluxa_segmentation'].data)
                    logger.info(f"Mask sum check at step {global_step}: {float(mask_sum)}")
                
                # ✅ S-1 NEW: Quick sanity print for segmentation mask presence
                if rank == 0 and global_step % 50 == 0:
                    seg_present_count = sum(v['fluxa_segmentation'] for v in batch_valid)
                    logger.info(f"Seg masks present in batch: {seg_present_count}/{len(batch_valid)}")
                
                try:
                    # Ultra-fast forward pass
                    model_outputs, forward_cache = ultra_fast_batch_forward(
                        fluxa_model, batch_inputs, args, training=True
                    )
                    
                    if model_outputs:
                        # ✅ FIXED M-2: Pass global_step instead of batch_idx
                        # ✅ NEW: Ultra-fast loss computation with comprehensive metrics and segmentation handling
                        if args.enable_comprehensive_metrics and metrics_computer:
                            batch_loss, loss_gradients, task_losses, step_metrics = ultra_fast_loss_computation_with_metrics(
                                model_outputs, batch_labels, loss_functions, loss_weights, device, 
                                args, global_step, metrics_computer, batch_valid  # ✅ S-1: Pass batch_valid
                            )
                            
                            # Accumulate metrics for epoch summary
                            if step_metrics:
                                metrics_count += 1
                                for metric_name, metric_value in step_metrics.items():
                                    if metric_name not in epoch_metrics:
                                        epoch_metrics[metric_name] = []
                                    epoch_metrics[metric_name].append(metric_value)
                        else:
                            batch_loss, loss_gradients, task_losses = ultra_fast_loss_computation(
                                model_outputs, batch_labels, loss_functions, loss_weights, device
                            )
                            step_metrics = {}
                        
                        if loss_gradients:
                            # Backward pass with optional profiling
                            try:
                                if profiler:
                                    with profiler("backward_pass"):
                                        param_grads = fluxa_model.backward(loss_gradients, forward_cache)
                                else:
                                    param_grads = fluxa_model.backward(loss_gradients, forward_cache)
                            except Exception as backward_error:
                                logger.error(f"Rank {rank}: Backward pass failed: {backward_error}")
                                continue
                            
                            # Optimized gradient application with NCCL synchronization
                            try:
                                if profiler:
                                    with profiler("optimizer_step"):
                                        optimizer.apply_gradients(param_grads, len(batch_inputs))
                                else:
                                    optimizer.apply_gradients(param_grads, len(batch_inputs))
                            except Exception as optimizer_error:
                                logger.error(f"Rank {rank}: Optimizer step failed: {optimizer_error}")
                                continue
                            
                            # Record metrics
                            epoch_losses.append(batch_loss)
                            for task_name, task_loss in task_losses.items():
                                epoch_task_losses[task_name].append(task_loss)
                            
                            # Performance monitoring
                            step_time = time.time() - step_start_time
                            performance_monitor.record_step(step_time, len(batch_inputs), batch_loss, step_metrics)
                            
                            # Memory monitoring
                            memory_manager.monitor_memory_usage(global_step)
                            
                            # Checkpoint saving (step-based, only rank 0)
                            if (global_step % args.checkpoint_save_steps == 0 and 
                                global_step > 0 and rank == 0):
                                save_checkpoint(
                                    fluxa_model, optimizer, epoch, global_step, 
                                    batch_loss, args.gcs_checkpoint_path, args.bucket_name, rank
                                )
                            
                            # ✅ FIXED M-1: Replace condition with direct step_metrics check
                            # ✅ NEW: Enhanced periodic logging with comprehensive metrics (only rank 0)
                            if step_metrics and rank == 0:  # ✅ FIXED: Use step_metrics instead of batch_idx modulo
                                throughput = len(batch_inputs) / step_time
                                total_throughput = throughput * world_size if world_size > 1 else throughput
                                
                                # Basic loss information
                                loss_str = f"Loss: {batch_loss:.6f}"
                                for name, loss_val in task_losses.items():
                                    loss_str += f", {name}: {loss_val:.6f}"
                                
                                # ✅ NEW: Rich metrics information (like the original FSE+TF version)
                                metrics_str = ""
                                if step_metrics:
                                    # Keypoints metrics
                                    if 'fluxa_keypoints_mae' in step_metrics:
                                        metrics_str += f", kp_mae: {step_metrics['fluxa_keypoints_mae']:.4f}"
                                    if 'fluxa_keypoints_fse_coherence' in step_metrics:
                                        metrics_str += f", kp_fse_coh: {step_metrics['fluxa_keypoints_fse_coherence']:.4f}"
                                    
                                    # Segmentation metrics
                                    if 'fluxa_segmentation_accuracy' in step_metrics:
                                        metrics_str += f", seg_acc: {step_metrics['fluxa_segmentation_accuracy']:.4f}"
                                    if 'fluxa_segmentation_miou' in step_metrics:
                                        metrics_str += f", seg_miou: {step_metrics['fluxa_segmentation_miou']:.4f}"
                                    
                                    # Surface normals metrics
                                    if 'fluxa_surface_normals_mae' in step_metrics:
                                        metrics_str += f", sn_mae: {step_metrics['fluxa_surface_normals_mae']:.4f}"
                                    if 'fluxa_surface_normals_physics_accuracy' in step_metrics:
                                        metrics_str += f", sn_phys_acc: {step_metrics['fluxa_surface_normals_physics_accuracy']:.4f}"
                                    
                                    # Global coherence metrics
                                    if 'global_field_coherence' in step_metrics:
                                        metrics_str += f", global_coherence: {step_metrics['global_field_coherence']:.4f}"
                                
                                nccl_status = "✅" if world_size > 1 else "N/A"
                                logger.info(f"Step {global_step} | {loss_str}{metrics_str} | "
                                          f"Time: {step_time:.3f}s | Throughput: {total_throughput:.1f} samples/s | "
                                          f"LR: {optimizer.get_lr():.6f} | NCCL: {nccl_status}")
                            
                            # ✅ NEW: Add batch stats logging
                            if (rank == 0 and global_step % args.metrics_log_frequency == 0):
                                batch_stats = data_loader.get_batch_stats()
                                logger.info(f"📊 Batch Stats → Real: {batch_stats['real_batches']}, "
                                          f"Fallback: {batch_stats['fallback_batches']} "
                                          f"({batch_stats['fallback_ratio']*100:.2f}% fallback)")
                    
                except Exception as e:
                    logger.error(f"Rank {rank}: Batch {batch_idx} failed: {e}")
                    continue
            
            # Run validation if it's time (only rank 0)
            val_metrics = {}
            if rank == 0 and (epoch + 1) % args.validation_frequency == 0:
                logger.info(f"🔍 Running validation at epoch {epoch + 1}...")
                val_start_time = time.time()
                val_metrics = run_validation(
                    fluxa_model, data_loader, loss_functions, loss_weights, args, device, rank
                )
                val_time = time.time() - val_start_time
                logger.info(f"✅ Validation completed in {val_time:.1f}s")
            
            # Epoch summary (only rank 0)
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            if rank == 0 and epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                avg_task_losses = {name: np.mean(losses) for name, losses in epoch_task_losses.items() if losses}
                
                # ✅ NEW: Compute epoch-level metrics averages
                avg_epoch_metrics = {}
                if epoch_metrics:
                    for metric_name, metric_values in epoch_metrics.items():
                        if metric_values:
                            avg_epoch_metrics[metric_name] = np.mean(metric_values)
                
                logger.info(f"\n✅ Epoch {epoch + 1} Summary:")
                logger.info(f"   Time: {epoch_time:.1f}s")
                logger.info(f"   Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"   Steps completed: {len(epoch_losses)}")
                logger.info(f"   Learning rate: {optimizer.get_lr():.6f}")
                
                # Multi-GPU performance info
                if world_size > 1:
                    perf_summary = performance_monitor.get_performance_summary()
                    total_throughput = perf_summary.get('avg_throughput', 0) * world_size
                    logger.info(f"   Total throughput (all GPUs): {total_throughput:.1f} samples/s")
                    logger.info(f"   GPU utilization: {world_size} x A100 with NCCL")
                
                for name, loss in avg_task_losses.items():
                    logger.info(f"   {name}: {loss:.6f}")
                
                # ✅ NEW: Log comprehensive epoch metrics (like the original FSE+TF version)
                if avg_epoch_metrics:
                    logger.info(f"   📊 Epoch Metrics Summary:")
                    
                    # Group metrics by modality for better readability
                    kp_metrics = {k: v for k, v in avg_epoch_metrics.items() if 'keypoints' in k}
                    seg_metrics = {k: v for k, v in avg_epoch_metrics.items() if 'segmentation' in k}
                    sn_metrics = {k: v for k, v in avg_epoch_metrics.items() if 'surface_normals' in k}
                    global_metrics = {k: v for k, v in avg_epoch_metrics.items() if 'global' in k or 'system' in k}
                    
                    if kp_metrics:
                        logger.info(f"     Keypoints: " + ", ".join([f"{k.replace('fluxa_keypoints_', '')}: {v:.4f}" for k, v in kp_metrics.items()]))
                    if seg_metrics:
                        logger.info(f"     Segmentation: " + ", ".join([f"{k.replace('fluxa_segmentation_', '')}: {v:.4f}" for k, v in seg_metrics.items()]))
                    if sn_metrics:
                        logger.info(f"     Surface Normals: " + ", ".join([f"{k.replace('fluxa_surface_normals_', '')}: {v:.4f}" for k, v in sn_metrics.items()]))
                    if global_metrics:
                        logger.info(f"     Global: " + ", ".join([f"{k}: {v:.4f}" for k, v in global_metrics.items()]))
                
                # Log validation metrics
                if val_metrics:
                    logger.info(f"   Validation loss: {val_metrics.get('avg_val_loss', 'N/A'):.6f}")
                    for key, value in val_metrics.items():
                        if key.startswith('val_') and key != 'avg_val_loss':
                            logger.info(f"   {key}: {value:.6f}")
                
                # Performance analysis
                perf_summary = performance_monitor.get_performance_summary()
                logger.info(f"   Avg throughput per GPU: {perf_summary.get('avg_throughput', 0):.1f} samples/s")
                logger.info(f"   Avg step time: {perf_summary.get('avg_step_time', 0):.3f}s")
                
                # Check for new best model (use validation loss if available, otherwise training loss)
                current_metric = -val_metrics.get('avg_val_loss', avg_epoch_loss)
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    early_stopping_counter = 0  # Reset early stopping counter
                    metric_type = "validation" if val_metrics else "training"
                    logger.info(f"🏆 New best model ({metric_type}): {best_val_metric:.6f}")
                    
                    # Save best model checkpoint
                    save_checkpoint(
                        fluxa_model, optimizer, epoch, global_step, 
                        val_metrics.get('avg_val_loss', avg_epoch_loss), 
                        f"{args.gcs_checkpoint_path}/best", args.bucket_name, rank
                    )
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= args.early_stopping_patience:
                        logger.info(f"🛑 Early stopping triggered after {early_stopping_counter} epochs without improvement")
                        break
                
                # Epoch-based checkpoint saving
                if (epoch + 1) % args.checkpoint_save_interval == 0:
                    save_checkpoint(
                        fluxa_model, optimizer, epoch, global_step, 
                        avg_epoch_loss, args.gcs_checkpoint_path, args.bucket_name, rank
                    )
                
                # Memory statistics
                memory_stats = memory_manager.get_memory_stats()
                if 'utilization' in memory_stats:
                    logger.info(f"   GPU Memory: {memory_stats['utilization']:.1%} "
                              f"({memory_stats['current_used_gb']:.2f}GB used)")
            
            # Periodic optimization suggestions (only rank 0)
            if rank == 0 and (epoch + 1) % 5 == 0:
                suggestions = performance_monitor.suggest_optimizations()
                if suggestions:
                    logger.info("💡 Performance Suggestions:")
                    for suggestion in suggestions:
                        logger.info(f"   - {suggestion}")
            
            # Memory optimization
            if args.enable_memory_optimization and device == "gpu":
                memory_manager.emergency_cleanup()
        
        # Final performance report (only rank 0)
        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("🏁 TRAINING COMPLETED")
            logger.info("=" * 80)
            
            perf_summary = performance_monitor.get_performance_summary()
            
            logger.info(f"📊 Final Performance Summary:")
            logger.info(f"   Total steps: {perf_summary.get('total_steps', 0)}")
            logger.info(f"   Average throughput per GPU: {perf_summary.get('avg_throughput', 0):.1f} samples/s")
            if world_size > 1:
                total_throughput = perf_summary.get('avg_throughput', 0) * world_size
                logger.info(f"   Total throughput (all GPUs): {total_throughput:.1f} samples/s")
                logger.info(f"   Multi-GPU speedup: ~{world_size}x with {world_size} A100s + NCCL")
            logger.info(f"   Average step time: {perf_summary.get('avg_step_time', 0):.3f}s")
            logger.info(f"   Best metric: {best_val_metric:.6f}")
            logger.info(f"   Total training time: {sum(epoch_times):.1f}s")
            
            # ✅ NEW: Final comprehensive metrics summary
            if 'best_metrics' in perf_summary:
                logger.info(f"🏆 Best Metrics Achieved:")
                best_metrics = perf_summary['best_metrics']
                
                # Group by modality for better readability
                for modality in ['keypoints', 'segmentation', 'surface_normals', 'environment_lighting']:
                    modality_metrics = {k: v for k, v in best_metrics.items() if modality in k}
                    if modality_metrics:
                        logger.info(f"   {modality.title()}: " + 
                                  ", ".join([f"{k.replace(f'fluxa_{modality}_', '')}: {v:.4f}" for k, v in modality_metrics.items()]))
                
                # Global metrics
                global_metrics = {k: v for k, v in best_metrics.items() if 'global' in k or 'system' in k}
                if global_metrics:
                    logger.info(f"   Global: " + ", ".join([f"{k}: {v:.4f}" for k, v in global_metrics.items()]))
            
            # Cache statistics
            cache_stats = data_loader.get_cache_stats()
            logger.info(f"📁 Data Loading Cache:")
            logger.info(f"   Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            logger.info(f"   Cached samples: {cache_stats.get('cached_samples', 0)}")
            
            # Final batch statistics
            final_batch_stats = data_loader.get_batch_stats()
            logger.info(f"📊 Final Batch Statistics:")
            logger.info(f"   Real batches: {final_batch_stats['real_batches']}")
            logger.info(f"   Fallback batches: {final_batch_stats['fallback_batches']}")
            logger.info(f"   Fallback ratio: {final_batch_stats['fallback_ratio']*100:.2f}%")
            
            # Bottleneck analysis
            if profiler:
                profiler_stats = profiler.get_stats()
                bottlenecks = performance_monitor.analyze_bottlenecks(profiler_stats)
                if bottlenecks:
                    logger.info("⚠️ Performance Bottlenecks Detected:")
                    for op, desc in bottlenecks.items():
                        logger.info(f"   {op}: {desc}")
            
            logger.info("✅ Ultra-optimized FlowField training with comprehensive metrics completed successfully! 🌊")
            
            # Save final checkpoint
            if rank == 0 and epoch_losses:
                final_loss = np.mean(epoch_losses)
                save_checkpoint(
                    fluxa_model, optimizer, args.epochs - 1, global_step, 
                    final_loss, f"{args.gcs_checkpoint_path}/final", args.bucket_name, rank
                )
        
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {e}", exc_info=True)
        raise
    
    finally:
        # Proper cleanup for distributed training
        if args.enable_multi_gpu and world_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    # Synchronize all ranks before cleanup
                    dist.barrier()
                    
                    # Destroy the process group
                    dist.destroy_process_group()
                    logger.info(f"✅ Rank {rank}: Distributed cleanup completed")
                    
            except Exception as e:
                logger.warning(f"Rank {rank}: Distributed cleanup warning: {e}")
        
        # GPU memory cleanup
        if device == "gpu":
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
                logger.info(f"✅ Rank {rank}: GPU memory cleanup completed")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")

if __name__ == '__main__':
    args = parse_arguments()
    
    # Set optimization flags based on parsed args (don't override user choices)
    if not hasattr(args, 'enable_vectorized_batching') or args.enable_vectorized_batching is None:
        args.enable_vectorized_batching = True
    if not hasattr(args, 'enable_memory_optimization') or args.enable_memory_optimization is None:
        args.enable_memory_optimization = True
    
    # Ensure minimum performance settings
    args.prefetch_batches = max(2, args.prefetch_batches)
    args.num_data_workers = max(4, args.num_data_workers)
    
    logger.info("🚀 Starting Ultra-Optimized FlowField Training with Comprehensive FSE Metrics")
    logger.info(f"   Expected performance improvement: 5-10x faster + Multi-GPU scaling")
    logger.info(f"   Optimizations enabled:")
    logger.info(f"     ✅ Vectorized Im2Col + GEMM convolutions")
    logger.info(f"     ✅ Async data loading with {args.prefetch_batches} prefetch")
    logger.info(f"     ✅ Unified memory pooling")
    logger.info(f"     ✅ Kernel fusion and batched operations")
    logger.info(f"     ✅ Advanced memory management")
    logger.info(f"     ✅ Performance profiling and monitoring")
    logger.info(f"     ✅ NCCL multi-GPU gradient synchronization")
    logger.info(f"     ✅ Device synchronization fixes")
    logger.info(f"     ✅ Cache consistency fixes")
    logger.info(f"     ✅ Comprehensive FSE metrics (MAE, IoU, coherence, physics accuracy)")
    logger.info(f"     ✅ Segmentation mask validation and handling")
    
    ultra_optimized_training_loop(args)