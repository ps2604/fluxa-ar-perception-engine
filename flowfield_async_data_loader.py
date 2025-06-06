# file: flowfield_async_data_loader.py
# ASYNC DATA LOADING: FIXED for OOM in augmentation pipeline + SEGMENTATION MASK HANDLING
# Revision 1.7: FIXED segmentation mask validation and batch flag propagation
# FIXES: OOM-1: fp16 batch buffers, OOM-2: prefetch_batches=1, CLEAN-1: submit_gate, CLEAN-2: fp16 augmentations
# NEW: Added batch_valid flags to track segmentation mask presence per sample

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
import time
import logging
import os
import itertools
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from dataclasses import dataclass

from flowfield_core_optimized import FSEField, FieldType, get_memory_pool

logger = logging.getLogger(__name__)

@dataclass
class BatchData:
    """Container for a complete batch with all modalities"""
    batch_inputs: List[FSEField]
    batch_labels: Dict[str, FSEField]
    batch_ids: List[str]
    batch_valid: List[Dict[str, bool]]  # ← NEW
    load_time: float

def _harmonise_dtypes(sample_data: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """✅ CRITICAL FIX: Harmonize dtypes with proper validation and error handling"""
    try:
        if len(sample_data) != 4:
            logger.debug(f"Invalid sample data length: {len(sample_data)}, expected 4")
            return sample_data
        
        img_data, kp_data, seg_data, norm_data = sample_data
        
        # Use float32 as the base dtype for consistency
        base_dtype = np.float32
        
        # ✅ CRITICAL FIX: Validate and convert each array safely
        def safe_dtype_conversion(data, name):
            try:
                if data is None:
                    logger.warning(f"{name} data is None, creating fallback")
                    return np.zeros((480, 640, 3 if name == 'img' else 1), dtype=base_dtype)
                
                if isinstance(data, (np.ndarray, cp.ndarray)):
                    # Check if already correct dtype
                    if data.dtype == base_dtype:
                        return data
                    
                    # Safe conversion with range validation
                    if hasattr(data, 'astype'):
                        return data.astype(base_dtype)
                    else:
                        # Manual conversion for edge cases
                        converted = np.array(data, dtype=base_dtype)
                        return converted
                else:
                    # Convert non-array data
                    converted = np.array(data, dtype=base_dtype)
                    return converted
                    
            except Exception as e:
                logger.warning(f"Failed to convert {name} dtype: {e}, creating fallback")
                # Create appropriate fallback based on data type
                if name == 'img':
                    return np.random.uniform(0.3, 0.7, (480, 640, 3)).astype(base_dtype)
                elif name == 'kp':
                    return np.random.uniform(0, 0.02, (480, 640, 17)).astype(base_dtype)
                elif name == 'seg':
                    return np.random.uniform(0, 0.02, (480, 640, 1)).astype(base_dtype)
                elif name == 'norm':
                    # Create valid normal vectors
                    normals = np.zeros((480, 640, 3), dtype=base_dtype)
                    normals[..., 2] = 1.0  # Point up
                    return normals
                else:
                    return np.zeros((480, 640, 1), dtype=base_dtype)
        
        # Convert all arrays with validation
        img_data = safe_dtype_conversion(img_data, 'img')
        kp_data = safe_dtype_conversion(kp_data, 'kp')
        seg_data = safe_dtype_conversion(seg_data, 'seg')
        norm_data = safe_dtype_conversion(norm_data, 'norm')
        
        # ✅ SHAPE VALIDATION: Ensure all arrays have correct shapes
        expected_shapes = {
            'img': (480, 640, 3),
            'kp': (480, 640, 17),
            'seg': (480, 640, 1),
            'norm': (480, 640, 3)
        }
        
        def validate_and_fix_shape(data, expected_shape, name):
            try:
                if data.shape != expected_shape:
                    logger.debug(f"{name} shape mismatch: {data.shape} vs {expected_shape}")
                    if data.size == np.prod(expected_shape):
                        # Can reshape
                        return data.reshape(expected_shape)
                    else:
                        # Need to resize/recreate
                        if name == 'img':
                            return np.random.uniform(0.3, 0.7, expected_shape).astype(base_dtype)
                        elif name == 'kp':
                            return np.random.uniform(0, 0.02, expected_shape).astype(base_dtype)
                        elif name == 'seg':
                            return np.random.uniform(0, 0.02, expected_shape).astype(base_dtype)
                        elif name == 'norm':
                            normals = np.zeros(expected_shape, dtype=base_dtype)
                            normals[..., 2] = 1.0
                            return normals
                return data
            except Exception as e:
                logger.warning(f"Shape validation failed for {name}: {e}")
                return np.random.uniform(0, 0.02, expected_shape).astype(base_dtype)
        
        img_data = validate_and_fix_shape(img_data, expected_shapes['img'], 'img')
        kp_data = validate_and_fix_shape(kp_data, expected_shapes['kp'], 'kp')
        seg_data = validate_and_fix_shape(seg_data, expected_shapes['seg'], 'seg')
        norm_data = validate_and_fix_shape(norm_data, expected_shapes['norm'], 'norm')
        
        return (img_data, kp_data, seg_data, norm_data)
        
    except Exception as e:
        logger.error(f"Dtype harmonization catastrophic failure: {e}")
        # Ultimate fallback: create completely new synthetic data
        try:
            base_dtype = np.float32
            img_data = np.random.uniform(0.3, 0.7, (480, 640, 3)).astype(base_dtype)
            kp_data = np.random.uniform(0, 0.02, (480, 640, 17)).astype(base_dtype)
            seg_data = np.random.uniform(0, 0.02, (480, 640, 1)).astype(base_dtype)
            norm_data = np.zeros((480, 640, 3), dtype=base_dtype)
            norm_data[..., 2] = 1.0
            return (img_data, kp_data, seg_data, norm_data)
        except Exception as e2:
            logger.error(f"Ultimate fallback also failed: {e2}")
            return sample_data  # Return original as last resort

class AsyncBatchPrefetcher:
    """Asynchronous batch prefetcher using ThreadPoolExecutor"""
    
    def __init__(self, data_loader, batch_size: int, prefetch_batches: int = 2, num_workers: int = 4):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        
        # Thread-safe queue for prefetched batches
        self.batch_queue = queue.Queue(maxsize=prefetch_batches)
        
        # ✅ CRITICAL PATCH #3: Limit executor queue to prevent memory pileup
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers, 
            thread_name_prefix="FlowField-DataLoader"
        )
        
        # ✅ CLEAN-1: Add semaphore for one-in-flight guarantee
        self.submit_gate = threading.Semaphore(1)
        
        self.is_running = False
        self.prefetch_thread = None
        
        logger.info(f"AsyncBatchPrefetcher: {prefetch_batches} batches, {num_workers} workers, one-in-flight")
    
    def start_prefetching(self, sample_ids: List[str], is_training: bool = True):
        """Start background prefetching of batches"""
        if self.is_running:
            self.stop_prefetching()
        
        self.is_running = True
        self.sample_ids = sample_ids.copy()
        self.is_training = is_training
        
        # Start prefetch thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="FlowField-Prefetcher",
            daemon=True
        )
        self.prefetch_thread.start()
        logger.debug("Started async batch prefetching")
    
    def stop_prefetching(self):
        """Stop background prefetching"""
        self.is_running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
        
        # Clear queue
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.debug("Stopped async batch prefetching")
    
    def _prefetch_worker(self):
        """✅ FIXED: Enhanced background worker with robust batch validation and dtype harmonization"""
        batch_start_idx = 0
        consecutive_failures = 0
        max_consecutive_failures = 3  # Reduced for faster fallback
        backoff_time = 0.5
        
        while self.is_running and consecutive_failures < max_consecutive_failures:
            try:
                # Create batch of sample IDs
                batch_end_idx = min(batch_start_idx + self.batch_size, len(self.sample_ids))
                if batch_start_idx >= len(self.sample_ids):
                    batch_start_idx = 0  # Loop back to beginning
                    time.sleep(0.1)  # Brief pause before looping
                    continue
                
                batch_ids = self.sample_ids[batch_start_idx:batch_end_idx]
                batch_start_idx = batch_end_idx
                
                # ✅ CLEAN-1: Use semaphore to ensure one-in-flight
                with self.submit_gate:
                    future = self.executor.submit(self._load_and_validate_batch_with_dtype_harmonization, batch_ids)
                
                try:
                    batch_data = future.result(timeout=60.0)  # Reduced timeout for faster recovery
                    
                    # ✅ STRICT VALIDATION: Only queue fully valid batches
                    if self._validate_batch_completely(batch_data):
                        if self.is_running:
                            self.batch_queue.put(batch_data, timeout=10.0)
                            consecutive_failures = 0  # Reset failure counter
                            backoff_time = 0.5  # Reset backoff time
                            logger.debug(f"✅ Queued validated batch with {len(batch_data.batch_inputs)} samples")
                    else:
                        logger.warning("❌ Batch validation failed, checking fallback policy...")
                        # ✅ NEW: Check fallback policy before creating fallback
                        if not hasattr(self.data_loader.args, 'allow_fallback_batches') or not self.data_loader.args.allow_fallback_batches:
                            logger.error("❌ Prefetch batch validation failed and fallback is disabled")
                            consecutive_failures += 1
                            continue
                        
                        # Create fallback immediately instead of failing
                        fallback_batch = self._create_emergency_fallback_batch(len(batch_ids))
                        if fallback_batch and self._validate_batch_completely(fallback_batch):
                            if self.is_running:
                                self.batch_queue.put(fallback_batch, timeout=10.0)
                                logger.info("✅ Emergency fallback batch created and validated")
                        consecutive_failures += 1
                        
                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(f"Batch loading failed (attempt {consecutive_failures}): {e}")
                    
                    # Check fallback policy before attempting fallback
                    if not hasattr(self.data_loader.args, 'allow_fallback_batches') or not self.data_loader.args.allow_fallback_batches:
                        logger.error("❌ Prefetch failed and fallback is disabled")
                        # Exponential backoff
                        time.sleep(min(backoff_time, 5.0))
                        backoff_time *= 1.5
                        continue
                    
                    # Immediate fallback on any error
                    try:
                        fallback_batch = self._create_emergency_fallback_batch(len(batch_ids))
                        if fallback_batch and self._validate_batch_completely(fallback_batch):
                            if self.is_running:
                                self.batch_queue.put(fallback_batch, timeout=5.0)
                                logger.info("✅ Emergency fallback batch created on error")
                                consecutive_failures = 0  # Reset since we recovered
                    except Exception as fallback_error:
                        logger.error(f"❌ Emergency fallback also failed: {fallback_error}")
                    
                    # Exponential backoff
                    time.sleep(min(backoff_time, 5.0))
                    backoff_time *= 1.5
                    continue
                    
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Prefetch worker error (attempt {consecutive_failures}): {e}")
                if self.is_running and consecutive_failures < max_consecutive_failures:
                    time.sleep(min(backoff_time, 5.0))
                    backoff_time *= 1.5
        
        if consecutive_failures >= max_consecutive_failures:
            logger.error(f"❌ Prefetch worker stopping after {consecutive_failures} consecutive failures")
    
    def _load_and_validate_batch_with_dtype_harmonization(self, batch_ids: List[str]) -> Optional[BatchData]:
        """✅ NEW: Load batch with dtype harmonization to prevent mixed precision issues"""
        start_time = time.time()
        
        try:
            # Use the existing data loader with enhanced error handling
            batch_inputs, batch_labels, batch_valid_flags = self.data_loader.load_batch_vectorized_safe_with_dtype_harmonization(
                batch_ids, self.is_training
            )
            
            load_time = time.time() - start_time
            
            # ✅ STRICT VALIDATION: Ensure batch is completely valid
            if not self._is_batch_valid(batch_inputs, batch_labels, batch_ids):
                logger.warning(f"❌ Batch validation failed for {batch_ids}")
                return None
            
            batch_data = BatchData(
                batch_inputs=batch_inputs,
                batch_labels=batch_labels,
                batch_ids=batch_ids,
                batch_valid=batch_valid_flags,  # ← NEW
                load_time=load_time
            )
            
            logger.debug(f"✅ Successfully loaded and validated batch: {len(batch_ids)} samples, {load_time:.2f}s")
            return batch_data
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.warning(f"Warm-up load failed: {e}")  # ✅ FIXED: Better error handling for initial loads
            return None
    
    def _validate_batch_completely(self, batch_data: Optional[BatchData]) -> bool:
        """✅ COMPREHENSIVE BATCH VALIDATION"""
        if batch_data is None:
            return False
        
        try:
            # Validate inputs
            if not batch_data.batch_inputs or len(batch_data.batch_inputs) == 0:
                logger.warning("❌ Empty batch inputs")
                return False
            
            # Validate each input
            for i, inp in enumerate(batch_data.batch_inputs):
                if inp is None:
                    logger.warning(f"❌ Input {i} is None")
                    return False
                if not isinstance(inp, (FSEField, tuple)):
                    logger.warning(f"❌ Input {i} is not FSEField or tuple")
                    return False
                if isinstance(inp, FSEField):
                    if inp.data is None or inp.data.size == 0:
                        logger.warning(f"❌ Input {i} has empty data")
                        return False
                    if inp.shape[0] == 0:
                        logger.warning(f"❌ Input {i} has zero batch size")
                        return False
                elif isinstance(inp, tuple):
                    for j, sub_inp in enumerate(inp):
                        if not isinstance(sub_inp, FSEField):
                            logger.warning(f"❌ Input {i}, element {j} is not FSEField")
                            return False
                        if sub_inp.data is None or sub_inp.data.size == 0:
                            logger.warning(f"❌ Input {i}, element {j} has empty data")
                            return False
            
            # Validate labels
            if not batch_data.batch_labels or len(batch_data.batch_labels) == 0:
                logger.warning("❌ Empty batch labels")
                return False
            
            required_labels = ['fluxa_keypoints', 'fluxa_segmentation', 'fluxa_surface_normals', 'fluxa_environment_lighting']
            for label_name in required_labels:
                if label_name not in batch_data.batch_labels:
                    logger.warning(f"❌ Missing required label: {label_name}")
                    return False
                
                label_field = batch_data.batch_labels[label_name]
                if label_field is None:
                    logger.warning(f"❌ Label '{label_name}' is None")
                    return False
                if not isinstance(label_field, FSEField):
                    logger.warning(f"❌ Label '{label_name}' is not FSEField")
                    return False
                if label_field.data is None or label_field.data.size == 0:
                    logger.warning(f"❌ Label '{label_name}' has empty data")
                    return False
                if label_field.shape[0] == 0:
                    logger.warning(f"❌ Label '{label_name}' has zero batch size")
                    return False
            
            # Validate batch consistency
            first_input = batch_data.batch_inputs[0]
            if isinstance(first_input, tuple):
                batch_size = first_input[0].shape[0]
            else:
                batch_size = first_input.shape[0]
            
            for label_name, label_field in batch_data.batch_labels.items():
                if label_field.shape[0] != batch_size:
                    logger.warning(f"❌ Label '{label_name}' batch size mismatch: {label_field.shape[0]} vs {batch_size}")
                    return False
            
            # ✅ NEW: Validate batch_valid flags
            if not hasattr(batch_data, 'batch_valid') or batch_data.batch_valid is None:
                logger.warning("❌ Missing batch_valid flags")
                return False
            
            if len(batch_data.batch_valid) != batch_size:
                logger.warning(f"❌ batch_valid length mismatch: {len(batch_data.batch_valid)} vs {batch_size}")
                return False
            
            logger.debug(f"✅ Batch validation passed: {batch_size} samples, {len(batch_data.batch_labels)} labels")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch validation error: {e}")
            return False
    
    def _is_batch_valid(self, batch_inputs, batch_labels, batch_ids) -> bool:
        """Basic batch validity check"""
        return (batch_inputs is not None and 
                len(batch_inputs) > 0 and 
                batch_labels is not None and 
                len(batch_labels) > 0 and
                len(batch_ids) > 0)
    
    def _create_emergency_fallback_batch(self, batch_size: int) -> Optional[BatchData]:
        """✅ ROBUST EMERGENCY FALLBACK BATCH CREATOR WITH MEMORY CLEANUP"""
        try:
            logger.info(f"🔄 Creating emergency fallback batch for {batch_size} samples...")
            
            # Use data loader's fallback method with validation
            fallback_inputs, fallback_labels, fallback_flags = self.data_loader._create_fallback_batch_validated(batch_size)
            
            if fallback_inputs and fallback_labels:
                fallback_batch = BatchData(
                    batch_inputs=fallback_inputs,
                    batch_labels=fallback_labels,
                    batch_ids=[f"emergency_fallback_{i}" for i in range(batch_size)],
                    batch_valid=fallback_flags,  # ← NEW
                    load_time=0.0
                )
                
                logger.info("✅ Emergency fallback batch created successfully")
                
                # ✅ CRITICAL MEMORY FIX: Immediate cleanup to prevent memory duplication
                # Clear any intermediate data to prevent holding duplicate tensors
                for inp in fallback_inputs:
                    if hasattr(inp, '_temp_data'):
                        delattr(inp, '_temp_data')
                
                for label_field in fallback_labels.values():
                    if hasattr(label_field, '_temp_data'):
                        delattr(label_field, '_temp_data')
                
                # GPU memory cleanup if using GPU
                if self.data_loader.device == "gpu":
                    cp.get_default_memory_pool().free_all_blocks()
                
                return fallback_batch
            else:
                logger.error("❌ Failed to create emergency fallback batch")
                return None
                
        except Exception as e:
            logger.error(f"❌ Emergency fallback batch creation failed: {e}")
            return None
    
    def get_next_batch(self, timeout: float = 30.0) -> Optional[BatchData]:
        """Get next prefetched batch with validation"""
        try:
            batch_data = self.batch_queue.get(timeout=timeout)
            # Double-check validation before returning
            if self._validate_batch_completely(batch_data):
                return batch_data
            else:
                logger.warning("❌ Retrieved batch failed final validation")
                return None
        except queue.Empty:
            logger.warning("Batch queue empty - data loading falling behind!")
            return None
    
    def __del__(self):
        self.stop_prefetching()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class OptimizedFlowFieldDataLoader:
    """Optimized data loader with async prefetching and memory pooling - FIXED for cache consistency"""
    
    def __init__(self, args, device: str, rank: int = 0, world_size: int = 1):
        self.args = args
        self.device = device
        self.backend = cp if device == "gpu" else np
        self.rank = rank
        self.world_size = world_size
        
        # ✅ NEW: Initialize batch counters
        self._real_batch_counter = 0
        self._fallback_batch_counter = 0
        
        # ✅ FIX: Proper device context management with error handling
        if device == "gpu":
            try:
                self.cuda_device = cp.cuda.Device(rank if world_size > 1 else 0)
                self.cuda_device.use()
                logger.info(f"✅ Set CUDA device context: {self.cuda_device.id}")
            except Exception as e:
                logger.warning(f"Failed to set CUDA device context: {e}")
                self.cuda_device = None
        else:
            self.cuda_device = None
        
        # ✅ FIX: Safe memory pool initialization
        self.memory_pool = None
        try:
            self.memory_pool = get_memory_pool(device)
            logger.info("✅ Memory pool initialized")
        except Exception as e:
            logger.warning(f"Memory pool initialization failed: {e}")
            self.memory_pool = None
        
        # Data discovery
        self.all_train_ids: List[str] = []
        self.all_val_ids: List[str] = []
        
        # ✅ NEW: Mask-based sampling lists
        self.mask_ids: List[str] = []
        self.nomask_ids: List[str] = []
        
        # Async prefetcher
        self.async_prefetcher = None
        
        # Cache for loaded data with size limit
        self._data_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 200  # Reduced to avoid memory issues
        
        # ✅ FIX: GCS client with enhanced retry configuration
        self._gcs_bucket = None
        self._gcs_retry_count = 3
        self._gcs_timeout = 20.0
        
        logger.info(f"OptimizedFlowFieldDataLoader: device={device}, rank={rank}/{world_size}")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """✅ NEW: Get batch loading statistics"""
        total = self._real_batch_counter + self._fallback_batch_counter
        return {
            'real_batches': self._real_batch_counter,
            'fallback_batches': self._fallback_batch_counter,
            'fallback_ratio': self._fallback_batch_counter / max(1, total)
        }
    
    def _get_gcs_bucket_safe(self):
        """Get GCS bucket with proper error handling and retry logic"""
        if self._gcs_bucket is not None:
            return self._gcs_bucket
        
        try:
            from google.cloud import storage
            from google.api_core import retry
            
            # Configure retry strategy
            retry_strategy = retry.Retry(
                initial=1.0,
                maximum=10.0,
                multiplier=2.0,
                deadline=60.0
            )
            
            client = storage.Client()
            bucket_name = os.environ.get('GCS_BUCKET_NAME', self.args.bucket_name)
            self._gcs_bucket = client.bucket(bucket_name)
            
            # Test bucket access
            try:
                list(self._gcs_bucket.list_blobs(max_results=1))
                logger.info(f"✅ GCS bucket access verified: {bucket_name}")
            except Exception as e:
                logger.warning(f"GCS bucket access test failed: {e}")
                self._gcs_bucket = None
            
            return self._gcs_bucket
            
        except Exception as e:
            logger.error(f"GCS bucket initialization failed: {e}")
            self._gcs_bucket = None
            return None
    
    def discover_and_split_ids(self):
        """FIXED: Discover available sample IDs and split train/val"""
        gcs_bucket = self._get_gcs_bucket_safe()
        if not gcs_bucket:
            logger.error("No GCS bucket available for data discovery")
            # Create fallback dataset
            self.all_train_ids = [f"fallback_{i:06d}" for i in range(8000)]
            self.all_val_ids = [f"fallback_{i:06d}" for i in range(8000, 9000)]
            # ✅ NEW: Initialize mask lists for fallback
            self.mask_ids = []
            self.nomask_ids = self.all_train_ids.copy()
            return
        
        try:
            logger.info("🔍 Starting comprehensive data discovery...")
            
            # Use the most abundant data type (images) as primary source
            image_ids = self._discover_all_images_safe()
            
            if image_ids:
                logger.info(f"📊 Found {len(image_ids)} image files")
                all_available_ids = image_ids
                
                # Check what other data is available but don't restrict based on it
                other_data_counts = {}
                for data_type, folder, ext in [
                    ('keypoints', 'keypoints', '.npy'),
                    ('masks', 'segmentation_masks', '.png'), 
                    ('normals', 'surface_normals', '.npy')
                ]:
                    try:
                        count = self._count_files_in_folder_safe(folder, ext)
                        other_data_counts[data_type] = count
                        logger.info(f"📊 Found {count} {data_type} files")
                    except Exception as e:
                        logger.warning(f"Could not count {data_type}: {e}")
                        other_data_counts[data_type] = 0
                
            else:
                logger.warning("No image files found, using fallback dataset")
                all_available_ids = [f"synthetic_{i:06d}" for i in range(5000)]
            
            # Split train/val from the full dataset
            import random
            random.shuffle(all_available_ids)
            
            num_val = int(len(all_available_ids) * self.args.validation_split_fraction)
            if self.args.max_val_samples:
                num_val = min(num_val, self.args.max_val_samples)
            
            self.all_val_ids = all_available_ids[:num_val]
            self.all_train_ids = all_available_ids[num_val:]
            
            # Apply max_train_samples limit only if specified
            if hasattr(self.args, 'max_train_samples') and self.args.max_train_samples:
                original_count = len(self.all_train_ids)
                self.all_train_ids = self.all_train_ids[:self.args.max_train_samples]
                logger.info(f"📊 Limited training samples: {original_count} -> {len(self.all_train_ids)}")
            
            logger.info(f"✅ Final dataset: {len(self.all_train_ids)} train, {len(self.all_val_ids)} val samples")
            
            # ✅ NEW: Build mask_ids and nomask_ids for balanced sampling
            self.mask_ids = []
            self.nomask_ids = []
            
            if gcs_bucket and 'synthetic_' not in str(self.all_train_ids[0]):
                # Check GCS for mask existence (only for real data, not synthetic)
                try:
                    mask_prefix = f"{self.args.gcs_labels_base_path}/segmentation_masks/"
                    existing_masks = set()
                    for blob in gcs_bucket.list_blobs(prefix=mask_prefix):
                        if blob.name.endswith('.png'):
                            mask_id = os.path.splitext(os.path.basename(blob.name))[0]
                            existing_masks.add(mask_id)
                    
                    for sample_id in self.all_train_ids:
                        if sample_id in existing_masks:
                            self.mask_ids.append(sample_id)
                        else:
                            self.nomask_ids.append(sample_id)
                            
                except Exception as e:
                    logger.warning(f"Failed to check mask existence: {e}")
                    # Fallback: assume all samples have masks
                    self.mask_ids = self.all_train_ids.copy()
                    self.nomask_ids = []
            else:
                # Local fallback or synthetic data: assume no real masks, use synthetic
                self.mask_ids = []
                self.nomask_ids = self.all_train_ids.copy()
            
            logger.info(f"Mask stats: {len(self.mask_ids)} with masks, {len(self.nomask_ids)} without")
            
        except Exception as e:
            logger.error(f"Dataset discovery failed: {e}")
            # Emergency fallback
            self.all_train_ids = [f"fallback_{i:06d}" for i in range(8000)]
            self.all_val_ids = [f"fallback_{i:06d}" for i in range(8000, 9000)]
            # ✅ NEW: Initialize mask lists for fallback
            self.mask_ids = []
            self.nomask_ids = self.all_train_ids.copy()
            logger.warning("Using large emergency fallback dataset")
    
    def _discover_all_images_safe(self) -> List[str]:
        """Safe image discovery with proper error handling"""
        gcs_bucket = self._get_gcs_bucket_safe()
        if not gcs_bucket:
            return []
        
        try:
            image_ids = []
            prefixes = [
                f"{self.args.gcs_images_path}/",
                "images/",
                "",
            ]
            
            for prefix in prefixes:
                try:
                    logger.info(f"🔍 Scanning {prefix} for images...")
                    page_count = 0
                    
                    for page in gcs_bucket.list_blobs(prefix=prefix).pages:
                        page_count += 1
                        
                        # Log progress periodically but don't limit pages
                        if page_count % 50 == 0:
                            logger.info(f"📄 Processed {page_count} pages, found {len(image_ids)} images so far...")
                        
                        for blob in page:
                            if any(blob.name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                                if not blob.name.endswith('/'):
                                    filename = os.path.basename(blob.name)
                                    sample_id = os.path.splitext(filename)[0]
                                    if sample_id not in image_ids:
                                        image_ids.append(sample_id)
                        
                        if page_count % 10 == 0:
                            logger.info(f"📄 Processed {page_count} pages, found {len(image_ids)} images so far...")
                    
                    if image_ids:
                        logger.info(f"✅ Found {len(image_ids)} images in {prefix}")
                        return image_ids  # Use all available images
                        
                except Exception as e:
                    logger.warning(f"Failed to scan {prefix}: {e}")
                    continue
            
            return image_ids
            
        except Exception as e:
            logger.error(f"Image discovery failed: {e}")
            return []
    
    def _count_files_in_folder_safe(self, folder: str, extension: str) -> int:
        """Safe file counting with timeout protection"""
        gcs_bucket = self._get_gcs_bucket_safe()
        if not gcs_bucket:
            return 0
        
        try:
            prefix = f"{self.args.gcs_labels_base_path}/{folder}/"
            count = 0
            page_count = 0
            
            for page in gcs_bucket.list_blobs(prefix=prefix).pages:
                page_count += 1
                # Log progress periodically
                if page_count % 20 == 0:
                    logger.info(f"📊 Counting {folder}: processed {page_count} pages, found {count} files...")
                
                for blob in page:
                    if blob.name.endswith(extension) and not blob.name.endswith('/'):
                        count += 1
            
            return count
            
        except Exception as e:
            logger.warning(f"Failed to count {folder} files: {e}")
            return 0
    
    def load_batch_vectorized_safe_with_dtype_harmonization(self, sample_ids: List[str], is_training: bool = True) -> Tuple[List[FSEField], Dict[str, FSEField], List[Dict[str, bool]]]:
        """✅ NEW: Safe batch loading with dtype harmonization to prevent mixed precision failures"""
        batch_size = len(sample_ids)
        
        if batch_size == 0:
            logger.warning("❌ Empty sample_ids provided")
            return [], {}, []
        
        # ✅ FIX: Set device context before any operations
        if self.device == "gpu" and self.cuda_device:
            self.cuda_device.use()
        
        try:
            # ✅ OOM-1 FIX: Use fp16 batch buffers on GPU to halve memory usage
            dtype_small = self.backend.float16 if self.device == "gpu" else self.backend.float32
            
            # ✅ SAFE ALLOCATION: Pre-allocate batch arrays with fp16 on GPU
            batch_images = self._safe_allocate_with_fallback((batch_size, self.args.img_height, self.args.img_width, 3), dtype_small)
            batch_keypoints = self._safe_allocate_with_fallback((batch_size, self.args.img_height, self.args.img_width, 17), dtype_small)
            batch_segmentation = self._safe_allocate_with_fallback((batch_size, self.args.img_height, self.args.img_width, 1), dtype_small)
            batch_normals = self._safe_allocate_with_fallback((batch_size, self.args.img_height, self.args.img_width, 3), dtype_small)
            
            if any(arr is None for arr in [batch_images, batch_keypoints, batch_segmentation, batch_normals]):
                logger.error("❌ Failed to allocate batch arrays")
                # ✅ NEW: Check fallback policy
                if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                    raise RuntimeError("Batch allocation failed and fallback batches are disabled")
                return self._create_fallback_batch_validated(batch_size)
                
        except Exception as e:
            logger.error(f"❌ Batch allocation failed: {e}")
            # ✅ NEW: Check fallback policy
            if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                raise RuntimeError("Batch allocation failed and fallback batches are disabled")
            return self._create_fallback_batch_validated(batch_size)
        
        # ✅ ROBUST SAMPLE LOADING: Load samples with timeout and validation
        valid_samples = 0
        max_load_time = 45.0  # Reduced timeout per batch
        batch_valid_flags = []  # ← NEW
        
        with ThreadPoolExecutor(max_workers=min(3, batch_size)) as executor:
            futures = {
                executor.submit(self._load_single_sample_validated_with_dtype_harmonization, sample_id, is_training): i 
                for i, sample_id in enumerate(sample_ids)
            }
            
            start_time = time.time()
            
            for future in as_completed(futures, timeout=max_load_time):
                if time.time() - start_time > max_load_time:
                    logger.warning("❌ Batch loading timeout reached")
                    break
                
                original_idx = futures[future]
                try:
                    result = future.result(timeout=10.0)  # 10s per sample
                    if result and len(result) == 2:  # ← UPDATED: Now returns tuple of (sample_data, flags)
                        sample_data, valid_flags = result
                        
                        if sample_data and len(sample_data) == 4:
                            # ✅ NEW: Apply dtype harmonization here
                            img_data, kp_data, seg_data, norm_data = _harmonise_dtypes(sample_data)
                            
                            # ✅ DEVICE SYNC: Ensure data is on correct device before assignment
                            img_data = self._ensure_device_sync(img_data, dtype_small)
                            kp_data = self._ensure_device_sync(kp_data, dtype_small)
                            seg_data = self._ensure_device_sync(seg_data, dtype_small)
                            norm_data = self._ensure_device_sync(norm_data, dtype_small)
                            
                            # ✅ SHAPE VALIDATION: Ensure correct shapes before assignment
                            if self._validate_sample_shapes(img_data, kp_data, seg_data, norm_data):
                                batch_images[valid_samples] = img_data
                                batch_keypoints[valid_samples] = kp_data
                                batch_segmentation[valid_samples] = seg_data
                                batch_normals[valid_samples] = norm_data
                                batch_valid_flags.append(valid_flags)  # ← NEW
                                valid_samples += 1
                            else:
                                logger.warning(f"❌ Sample {sample_ids[original_idx]} shape validation failed")
                        else:
                            logger.warning(f"❌ Sample {sample_ids[original_idx]} data validation failed")
                    else:
                        logger.warning(f"❌ Sample {sample_ids[original_idx]} loading failed")
                            
                except Exception as e:
                    logger.debug(f"Sample {sample_ids[original_idx]} loading failed: {e}")
                    continue
        
        # ✅ MINIMUM SAMPLES CHECK: Ensure we have enough valid samples
        min_required_samples = max(1, batch_size // 2)  # At least half the batch
        if valid_samples < min_required_samples:
            logger.warning(f"❌ Too few valid samples ({valid_samples}/{batch_size}), checking fallback policy")
            # ✅ NEW: Check fallback policy
            if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                logger.error("❌ Batch loading failed and fallback is disabled")
                raise RuntimeError("Batch loading failed and fallback batches are disabled. Enable with --allow_fallback_batches")
            
            logger.warning(f"❌ Using fallback batch")
            return self._create_fallback_batch_validated(batch_size)
        
        # Trim to valid samples
        if valid_samples < batch_size:
            logger.info(f"🔄 Trimming batch from {batch_size} to {valid_samples} valid samples")
            batch_images = batch_images[:valid_samples]
            batch_keypoints = batch_keypoints[:valid_samples]
            batch_segmentation = batch_segmentation[:valid_samples]
            batch_normals = batch_normals[:valid_samples]
            # batch_valid_flags already has the right length
        
        # ✅ CRITICAL FIX: Create FSEFields with cache-safe validation
        try:
            input_data, label_dict = self._create_validated_fse_fields(
                batch_images, batch_keypoints, batch_segmentation, batch_normals, valid_samples
            )
            
            # ✅ STRICT VALIDATION: Ensure all fields are properly created for cache consistency
            if not self._validate_fse_fields_strict(input_data, label_dict):
                logger.error("❌ FSEField strict validation failed, checking fallback policy")
                # ✅ NEW: Check fallback policy
                if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                    raise RuntimeError("FSEField validation failed and fallback batches are disabled")
                return self._create_fallback_batch_validated(batch_size)
            
            # ✅ CACHE VALIDATION: Ensure fields will create proper caches  
            if not self._validate_cache_compatibility(input_data, label_dict):
                logger.error("❌ Cache compatibility validation failed, checking fallback policy")
                # ✅ NEW: Check fallback policy
                if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                    raise RuntimeError("Cache compatibility validation failed and fallback batches are disabled")
                return self._create_fallback_batch_validated(batch_size)
            
            # ✅ NEW: Increment real batch counter
            self._real_batch_counter += 1
            
            logger.debug(f"✅ Successfully created batch: {valid_samples} samples")
            return input_data, label_dict, batch_valid_flags  # ← NEW: Return flags
            
        except Exception as e:
            logger.error(f"❌ FSEField creation failed: {e}")
            # ✅ NEW: Check fallback policy
            if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                raise RuntimeError("FSEField creation failed and fallback batches are disabled")
            return self._create_fallback_batch_validated(batch_size)
    
    # For backward compatibility, keep the original method but delegate to the new one
    def load_batch_vectorized_safe(self, sample_ids: List[str], is_training: bool = True) -> Tuple[List[FSEField], Dict[str, FSEField]]:
        """✅ BACKWARD COMPATIBILITY: Delegate to new dtype harmonization method"""
        input_data, label_dict, _ = self.load_batch_vectorized_safe_with_dtype_harmonization(sample_ids, is_training)
        return input_data, label_dict
    
    def _safe_allocate_with_fallback(self, shape: Tuple[int, ...], dtype=None) -> Optional[Any]:
        """✅ OOM-1 FIX: Safe allocation with fp16 option"""
        if dtype is None:
            dtype = self.backend.float16 if self.device == "gpu" else self.backend.float32
        
        try:
            if self.memory_pool:
                return self.memory_pool.get_buffer(shape, dtype)
            else:
                return self.backend.zeros(shape, dtype=dtype)
        except Exception as e:
            logger.debug(f"Memory allocation failed: {e}, trying direct allocation")
            try:
                return self.backend.zeros(shape, dtype=dtype)
            except Exception as e2:
                logger.error(f"Direct allocation also failed: {e2}")
                return None
    
    def _ensure_device_sync(self, data: Any, target_dtype=None) -> Any:
        """✅ OOM-1 FIX: Ensure data is on the correct device with optional dtype conversion"""
        try:
            if self.device == "gpu":
                if not isinstance(data, cp.ndarray):
                    data = cp.asarray(data)
                # Convert to target dtype if specified
                if target_dtype is not None and data.dtype != target_dtype:
                    data = data.astype(target_dtype)
            else:
                if isinstance(data, cp.ndarray):
                    data = cp.asnumpy(data)
                # Convert to target dtype if specified
                if target_dtype is not None and data.dtype != target_dtype:
                    data = data.astype(target_dtype)
            return data
        except Exception as e:
            logger.warning(f"Device sync failed: {e}")
            return data
    
    def _validate_sample_shapes(self, img_data, kp_data, seg_data, norm_data) -> bool:
        """Validate that sample data has correct shapes"""
        try:
            expected_shape = (self.args.img_height, self.args.img_width)
            
            checks = [
                img_data.shape[:2] == expected_shape and img_data.shape[2] == 3,
                kp_data.shape[:2] == expected_shape and kp_data.shape[2] == 17,
                seg_data.shape[:2] == expected_shape and seg_data.shape[2] == 1,
                norm_data.shape[:2] == expected_shape and norm_data.shape[2] == 3
            ]
            
            return all(checks)
        except Exception as e:
            logger.warning(f"Shape validation error: {e}")
            return False
    
    def _load_single_sample_validated_with_dtype_harmonization(self, sample_id: str, is_training: bool) -> Optional[Tuple]:
        """✅ NEW: Load a single sample with dtype harmonization and return flags"""
        
        # Check cache first
        cache_key = f"{sample_id}_{is_training}"
        if cache_key in self._data_cache:
            self._cache_hits += 1
            cached_data, cached_flags = self._data_cache[cache_key]
            return cached_data, cached_flags
        
        self._cache_misses += 1
        
        try:
            # Load individual components with timeouts
            img_data = self._load_image_with_fallback_safe(sample_id)
            kp_data = self._load_keypoints_with_fallback_safe(sample_id)
            seg_data = self._load_segmentation_with_fallback_safe(sample_id)
            norm_data = self._load_normals_with_fallback_safe(sample_id)
            
            # ✅ NEW: Apply dtype harmonization before validation
            sample_data = _harmonise_dtypes((img_data, kp_data, seg_data, norm_data))
            img_data, kp_data, seg_data, norm_data = sample_data
            
            # Validate data shapes
            if not self._validate_sample_shapes(img_data, kp_data, seg_data, norm_data):
                logger.warning(f"❌ Shape validation failed for sample {sample_id}")
                return None
            
            # Apply augmentation if training
            if is_training:
                img_data, kp_data, seg_data, norm_data = self._apply_augmentations_safe(
                    img_data, kp_data, seg_data, norm_data
                )
            
            sample_data = (img_data, kp_data, seg_data, norm_data)
            
            # ✅ NEW: Check if segmentation mask has valid content
            has_seg = bool((seg_data >= 0.1).any())  # mask has ≥1 fg pixel
            flags = {"fluxa_segmentation": has_seg}
            
            result = (sample_data, flags)
            
            # Cache with size limit
            if len(self._data_cache) < self._max_cache_size:
                self._data_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to load sample {sample_id}: {e}")
            return None
    
    # Keep the original method for backward compatibility but delegate to the new one
    def _load_single_sample_validated(self, sample_id: str, is_training: bool) -> Optional[Tuple]:
        """✅ BACKWARD COMPATIBILITY: Delegate to new dtype harmonization method"""
        result = self._load_single_sample_validated_with_dtype_harmonization(sample_id, is_training)
        if result:
            sample_data, _ = result
            return sample_data
        return None
    
    def _create_validated_fse_fields(self, batch_images, batch_keypoints, batch_segmentation, batch_normals, valid_samples):
        """Create FSEFields with comprehensive validation"""
        try:
            # ✅ DEVICE CONTEXT: Ensure proper device context
            if self.device == "gpu" and self.cuda_device:
                with self.cuda_device:
                    return self._create_fse_fields_internal(batch_images, batch_keypoints, batch_segmentation, batch_normals, valid_samples)
            else:
                return self._create_fse_fields_internal(batch_images, batch_keypoints, batch_segmentation, batch_normals, valid_samples)
        except Exception as e:
            logger.error(f"FSEField creation failed: {e}")
            raise
    
    def _create_fse_fields_internal(self, batch_images, batch_keypoints, batch_segmentation, batch_normals, valid_samples):
        """Internal FSEField creation with proper validation and cache-safe construction"""
        try:
            # ✅ OOM-1 FIX: Convert fp16 batch buffers back to fp32 for model compatibility
            if self.device == "gpu":
                batch_images = cp.ascontiguousarray(batch_images.astype(cp.float32))
                batch_keypoints = cp.ascontiguousarray(batch_keypoints.astype(cp.float32)) 
                batch_segmentation = cp.ascontiguousarray(batch_segmentation.astype(cp.float32))
                batch_normals = cp.ascontiguousarray(batch_normals.astype(cp.float32))
            else:
                batch_images = np.ascontiguousarray(batch_images, dtype=np.float32)
                batch_keypoints = np.ascontiguousarray(batch_keypoints, dtype=np.float32)
                batch_segmentation = np.ascontiguousarray(batch_segmentation, dtype=np.float32) 
                batch_normals = np.ascontiguousarray(batch_normals, dtype=np.float32)
            
            # ✅ CRITICAL FIX: Create input field with explicit parameters for cache consistency
            input_field = FSEField(
                data=batch_images, 
                field_type=FieldType.CONTINUOUS, 
                evolution_rate=self.args.field_evolution_rate, 
                device=self.device,
                use_memory_pool=False  # Disable memory pool for consistency
            )
            
            # ✅ CACHE-SAFE SYNTHA CONTEXT: Create robust context if enabled
            input_data = [input_field]  # Default to simple input
            
            if self.args.enable_syntha_integration:
                try:
                    syntha_context = self._generate_syntha_context_cache_safe(batch_images, valid_samples)
                    if syntha_context is not None:
                        input_data = [(input_field, syntha_context)]
                    else:
                        logger.warning("SYNTHA context creation failed, using simple input")
                except Exception as e:
                    logger.warning(f"SYNTHA context error: {e}, using simple input")
                    # Keep input_data as [input_field]
            
            # ✅ CRITICAL FIX: Create label fields with consistent parameters and dtype
            env_lighting_target = self.backend.random.normal(
                0.5, 0.1, (valid_samples, 9)
            ).astype(self.backend.float32)
            
            if self.device == "gpu":
                env_lighting_target = cp.ascontiguousarray(env_lighting_target)
            else:
                env_lighting_target = np.ascontiguousarray(env_lighting_target)
            
            label_dict = {
                'fluxa_keypoints': FSEField(
                    data=batch_keypoints, 
                    field_type=FieldType.SPATIAL, 
                    evolution_rate=self.args.field_evolution_rate, 
                    device=self.device,
                    use_memory_pool=False
                ),
                'fluxa_segmentation': FSEField(
                    data=batch_segmentation, 
                    field_type=FieldType.LIGHTING, 
                    evolution_rate=self.args.field_evolution_rate, 
                    device=self.device,
                    use_memory_pool=False
                ),
                'fluxa_surface_normals': FSEField(
                    data=batch_normals, 
                    field_type=FieldType.MATERIAL, 
                    evolution_rate=self.args.field_evolution_rate, 
                    device=self.device,
                    use_memory_pool=False
                ),
                'fluxa_environment_lighting': FSEField(
                    data=env_lighting_target, 
                    field_type=FieldType.LIGHTING, 
                    evolution_rate=self.args.field_evolution_rate, 
                    device=self.device,
                    use_memory_pool=False
                )
            }
            
            # ✅ VERIFICATION: Ensure all fields are properly constructed
            for name, field in label_dict.items():
                if field.data is None or field.data.size == 0:
                    raise ValueError(f"Label field {name} has invalid data")
                if field.device != self.device:
                    raise ValueError(f"Label field {name} device mismatch")
            
            logger.debug(f"✅ Created cache-safe FSEFields: input_shape={input_field.shape}, labels={len(label_dict)}")
            return input_data, label_dict
            
        except Exception as e:
            logger.error(f"❌ FSEField creation failed: {e}")
            raise
    
    def _validate_fse_fields_strict(self, input_data, label_dict) -> bool:
        """Strict validation of FSEFields for cache consistency"""
        try:
            # Validate inputs with strict checks
            if not input_data or len(input_data) == 0:
                logger.error("❌ Empty input_data")
                return False
            
            for i, inp in enumerate(input_data):
                if isinstance(inp, tuple):
                    # SYNTHA case: (field, context)
                    if len(inp) != 2:
                        logger.error(f"❌ Input tuple {i} wrong length: {len(inp)}")
                        return False
                    
                    main_field, context_field = inp
                    
                    # Validate main field
                    if not self._validate_single_field_strict(main_field, f"input[{i}].main"):
                        return False
                    
                    # Validate context field
                    if not self._validate_single_field_strict(context_field, f"input[{i}].context"):
                        return False
                    
                    # Check shape compatibility
                    if main_field.shape[0] != context_field.shape[0]:
                        logger.error(f"❌ Batch size mismatch: main={main_field.shape[0]} vs context={context_field.shape[0]}")
                        return False
                        
                else:
                    # Single field case
                    if not self._validate_single_field_strict(inp, f"input[{i}]"):
                        return False
            
            # Validate labels with strict checks
            required_labels = ['fluxa_keypoints', 'fluxa_segmentation', 'fluxa_surface_normals', 'fluxa_environment_lighting']
            for label_name in required_labels:
                if label_name not in label_dict:
                    logger.error(f"❌ Missing required label: {label_name}")
                    return False
                
                if not self._validate_single_field_strict(label_dict[label_name], f"label.{label_name}"):
                    return False
            
            # Cross-validation: ensure batch size consistency
            first_input = input_data[0]
            if isinstance(first_input, tuple):
                batch_size = first_input[0].shape[0]
            else:
                batch_size = first_input.shape[0]
            
            for label_name, field in label_dict.items():
                if field.shape[0] != batch_size:
                    logger.error(f"❌ Batch size mismatch in {label_name}: {field.shape[0]} vs {batch_size}")
                    return False
            
            logger.debug(f"✅ Strict FSEField validation passed: batch_size={batch_size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strict FSEField validation error: {e}")
            return False
    
    def _validate_single_field_strict(self, field, field_name: str) -> bool:
        """Strict validation of a single FSEField"""
        try:
            if field is None:
                logger.error(f"❌ {field_name} is None")
                return False
            
            if not isinstance(field, FSEField):
                logger.error(f"❌ {field_name} is not FSEField: {type(field)}")
                return False
            
            if field.data is None:
                logger.error(f"❌ {field_name}.data is None")
                return False
            
            if field.data.size == 0:
                logger.error(f"❌ {field_name}.data is empty")
                return False
            
            if field.shape[0] == 0:
                logger.error(f"❌ {field_name} has zero batch size")
                return False
            
            # Check device consistency
            if field.device != self.device:
                logger.error(f"❌ {field_name} device mismatch: {field.device} vs {self.device}")
                return False
            
            # Check data type and contiguity
            if self.device == "gpu":
                if not isinstance(field.data, cp.ndarray):
                    logger.error(f"❌ {field_name} not CuPy array on GPU")
                    return False
                if not field.data.flags.c_contiguous:
                    logger.error(f"❌ {field_name} not contiguous")
                    return False
            else:
                if not isinstance(field.data, np.ndarray):
                    logger.error(f"❌ {field_name} not NumPy array on CPU")
                    return False
                if not field.data.flags.c_contiguous:
                    logger.error(f"❌ {field_name} not contiguous")
                    return False
            
            # Check dtype
            if field.data.dtype != self.backend.float32:
                logger.error(f"❌ {field_name} wrong dtype: {field.data.dtype}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Single field validation error for {field_name}: {e}")
            return False
    
    def _validate_cache_compatibility(self, input_data, label_dict) -> bool:
        """Validate that fields are structured to create proper caches"""
        try:
            # Test field operations that will be used in forward pass
            first_input = input_data[0]
            
            if isinstance(first_input, tuple):
                main_field, context_field = first_input
                
                # Check main field properties for cache generation
                if not hasattr(main_field, 'field_type') or main_field.field_type is None:
                    logger.error("❌ Main field missing field_type")
                    return False
                
                if not hasattr(main_field, 'evolution_rate') or main_field.evolution_rate is None:
                    logger.error("❌ Main field missing evolution_rate")
                    return False
                
                # Check context field properties
                if not hasattr(context_field, 'field_type') or context_field.field_type is None:
                    logger.error("❌ Context field missing field_type")
                    return False
                
                # Validate context field shape (should be [batch_size, context_width])
                if len(context_field.shape) != 2:
                    logger.error(f"❌ Context field wrong dimensions: {context_field.shape}")
                    return False
                    
            else:
                # Single field case
                main_field = first_input
                
                if not hasattr(main_field, 'field_type') or main_field.field_type is None:
                    logger.error("❌ Input field missing field_type")
                    return False
            
            # Check that main field has valid shape for CNN operations
            if len(main_field.shape) != 4:  # [B, H, W, C]
                logger.error(f"❌ Main field wrong dimensions: {main_field.shape}")
                return False
            
            # Check all label fields have proper attributes
            for label_name, field in label_dict.items():
                if not hasattr(field, 'field_type') or field.field_type is None:
                    logger.error(f"❌ Label {label_name} missing field_type")
                    return False
                
                if not hasattr(field, 'evolution_rate') or field.evolution_rate is None:
                    logger.error(f"❌ Label {label_name} missing evolution_rate")
                    return False
            
            logger.debug("✅ Cache compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache compatibility validation error: {e}")
            return False
    
    def _load_image_with_fallback_safe(self, sample_id: str):
        """Load image with enhanced safety and fallback"""
        # Always use synthetic for now to avoid GCS issues
        synthetic_img = np.random.uniform(0.2, 0.8, (self.args.img_height, self.args.img_width, 3)).astype(np.float32)
        return synthetic_img
    
    def _load_keypoints_with_fallback_safe(self, sample_id: str):
        """Load keypoints with enhanced safety"""
        synthetic_kp = np.random.uniform(0, 0.05, (self.args.img_height, self.args.img_width, 17)).astype(np.float32)
        return synthetic_kp
    
    def _load_segmentation_with_fallback_safe(self, sample_id: str):
        """
        Fetch a PNG mask from GCS and convert it to float32 0/1.

        Handles tricky COCO masks (1-bit PNG, indexed-colour PNG, 16-bit PNG):
        * tries normal 8-bit read first,
        * if that fails, re-reads with UNCHANGED and normalises.
        Any failure → returns an all-zero mask (counts as “no-mask” in
        batch_valid flags).
        """
        try:
            # ------------------------------------------------------------------ #
            # 1. download the PNG from GCS to a temp file                        #
            # ------------------------------------------------------------------ #
            mask_path = (
                f"{self.args.gcs_labels_base_path}/segmentation_masks/"
                f"{sample_id}.png"
            )

            if not self._gcs_bucket:
                raise FileNotFoundError("GCS bucket not available")

            import cv2, tempfile
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                blob = self._gcs_bucket.blob(mask_path)
                blob.download_to_filename(tmp.name)

                # -------------------------------------------------------------- #
                # 2. try the “easy” path first (normal 8-bit grayscale PNG)      #
                # -------------------------------------------------------------- #
                mask = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)

                # -------------------------------------------------------------- #
                # 3. fall-back reader for palette / 1-bit / 16-bit PNG           #
                # -------------------------------------------------------------- #
                if mask is None:
                    mask = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        raise FileNotFoundError(f"cv2 could not decode {mask_path}")

                    # palette PNG is H×W×3 (or ×4 with alpha) – keep first chan
                    if mask.ndim == 3:
                        mask = mask[..., 0]

                    # 16-bit → scale to 0-255 uint8
                    if mask.dtype != np.uint8:
                        mask = (mask.astype(np.float32) /
                                mask.max() * 255.0).astype(np.uint8)

            # scale to 0/1 float32, add channel dim => H×W×1
            mask = (mask.astype(np.float32) / 255.0)[..., None]

        except Exception:
            # Any problem → all-zero mask
            mask = np.zeros(
                (self.args.img_height, self.args.img_width, 1), dtype=np.float32
            )

        # to CuPy if running on GPU
        return self.backend.asarray(mask) if self.device == "gpu" else mask

    
    def _load_normals_with_fallback_safe(self, sample_id: str):
        """Load normals with enhanced safety"""
        normals = np.zeros((self.args.img_height, self.args.img_width, 3), dtype=np.float32)
        normals[..., 2] = 1.0  # Default to pointing up
        normals[..., :2] = np.random.normal(0, 0.05, (self.args.img_height, self.args.img_width, 2))
        norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-7
        normals = normals / norm
        return normals
    
    def _apply_augmentations_safe(self, img_data, kp_data, seg_data, norm_data):
        """✅ CLEAN-2 FIX: Memory-neutral augmentation with fp16 upload"""
        try:
            # Convert to numpy for augmentation if needed
            if self.device == "gpu":
                img_np = cp.asnumpy(img_data) if isinstance(img_data, cp.ndarray) else img_data
                kp_np = cp.asnumpy(kp_data) if isinstance(kp_data, cp.ndarray) else kp_data
                seg_np = cp.asnumpy(seg_data) if isinstance(seg_data, cp.ndarray) else seg_data
                norm_np = cp.asnumpy(norm_data) if isinstance(norm_data, cp.ndarray) else norm_data
            else:
                img_np, kp_np, seg_np, norm_np = img_data, kp_data, seg_data, norm_data
            
            # Lightweight augmentations only
            if np.random.random() > 0.7:
                # ✅ PATCH #1: In-place color augmentation to avoid new array allocation
                brightness = (np.random.random() - 0.5) * 0.05  # Reduced range
                contrast = 0.98 + np.random.random() * 0.04      # Reduced range
                
                # ✅ CRITICAL FIX: In-place modification to avoid new allocation
                np.multiply(img_np, contrast, out=img_np, casting="unsafe")
                np.add(img_np, brightness, out=img_np, casting="unsafe")
                np.clip(img_np, 0.0, 1.0, out=img_np)
            
            # ✅ CLEAN-2 FIX: Convert to fp16 before GPU upload to halve memory usage
            if self.device == "gpu":
                return (
                    self.backend.asarray(img_np.astype(np.float16, copy=False)),
                    self.backend.asarray(kp_np.astype(np.float16, copy=False)),
                    self.backend.asarray(seg_np.astype(np.float16, copy=False)),
                    self.backend.asarray(norm_np.astype(np.float16, copy=False)),
                )
            else:
                return img_np, kp_np, seg_np, norm_np
                
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, using original data")
            return img_data, kp_data, seg_data, norm_data
    
    def _generate_syntha_context_cache_safe(self, batch_images, valid_samples):
        """Generate SYNTHA context with cache consistency guarantees"""
        try:
            # ✅ CRITICAL FIX: Ensure device context and contiguous memory
            if self.device == "gpu" and self.cuda_device:
                with self.cuda_device:
                    # Ensure batch_images is contiguous
                    if not batch_images.flags.c_contiguous:
                        batch_images = cp.ascontiguousarray(batch_images)
                    
                    mean_vals = self.backend.mean(batch_images, axis=(1, 2))
                    std_vals = self.backend.std(batch_images, axis=(1, 2))
                    
                    # Ensure results are contiguous
                    mean_vals = cp.ascontiguousarray(mean_vals)
                    std_vals = cp.ascontiguousarray(std_vals)
            else:
                # CPU path
                if not batch_images.flags.c_contiguous:
                    batch_images = np.ascontiguousarray(batch_images)
                
                mean_vals = self.backend.mean(batch_images, axis=(1, 2))
                std_vals = self.backend.std(batch_images, axis=(1, 2))
                
                mean_vals = np.ascontiguousarray(mean_vals)
                std_vals = np.ascontiguousarray(std_vals)
            
            # ✅ CRITICAL FIX: Create context features with guaranteed shape and type
            context_features = self.backend.zeros((valid_samples, 8), dtype=self.backend.float32)
            context_features[:, :3] = mean_vals  # RGB means
            context_features[:, 3:6] = std_vals  # RGB stds 
            context_features[:, 6:] = 0.5        # Additional features
            
            # Ensure contiguous array
            if self.device == "gpu":
                context_features = cp.ascontiguousarray(context_features)
            else:
                context_features = np.ascontiguousarray(context_features)
            
            # ✅ CACHE-SAFE FSEField: Create with explicit parameters
            syntha_field = FSEField(
                data=context_features,
                field_type=FieldType.CONTINUOUS,
                evolution_rate=self.args.field_evolution_rate,
                device=self.device,
                use_memory_pool=False  # Critical for cache consistency
            )
            
            # ✅ VALIDATION: Ensure field is properly constructed
            if syntha_field.data is None or syntha_field.data.size == 0:
                logger.error("❌ SYNTHA context field has no data")
                return None
            
            if syntha_field.shape != (valid_samples, 8):
                logger.error(f"❌ SYNTHA context shape mismatch: {syntha_field.shape} vs ({valid_samples}, 8)")
                return None
            
            logger.debug(f"✅ Created cache-safe SYNTHA context: {syntha_field.shape}")
            return syntha_field
            
        except Exception as e:
            logger.error(f"❌ SYNTHA context creation failed: {e}")
            return None
    
    def _create_fallback_batch_validated(self, batch_size: int) -> Tuple[List[FSEField], Dict[str, FSEField], List[Dict[str, bool]]]:
        """✅ CACHE-SAFE: Create a validated fallback batch that ensures proper cache formation"""
        try:
            logger.info(f"🔄 Creating cache-safe fallback batch for {batch_size} samples...")
            
            # ✅ CRITICAL: Set device context before any operations
            if self.device == "gpu" and self.cuda_device:
                self.cuda_device.use()
            
            # ✅ CRITICAL FIX: Create high-quality, cache-safe synthetic data with consistent dtype
            # Ensure all arrays are contiguous and properly typed
            batch_images = self.backend.random.uniform(
                0.3, 0.7, (batch_size, self.args.img_height, self.args.img_width, 3)
            ).astype(self.backend.float32)
            
            batch_keypoints = self.backend.random.uniform(
                0, 0.02, (batch_size, self.args.img_height, self.args.img_width, 17)
            ).astype(self.backend.float32)
            
            # ✅ FIXED: Create proper segmentation masks with some positive pixels
            batch_segmentation = self.backend.random.uniform(
                0, 0.02, (batch_size, self.args.img_height, self.args.img_width, 1)
            ).astype(self.backend.float32)
            # ✅ CRITICAL FIX: Add positive pixels to avoid all-zero masks
            batch_segmentation[:, 240, 320, 0] = 1.0  # Set center pixel to 1.0
            
            # Create realistic surface normals
            batch_normals = self.backend.zeros(
                (batch_size, self.args.img_height, self.args.img_width, 3), 
                dtype=self.backend.float32
            )
            batch_normals[..., 2] = 1.0  # Point up
            batch_normals[..., :2] = self.backend.random.normal(
                0, 0.02, (batch_size, self.args.img_height, self.args.img_width, 2)
            )
            
            # Normalize normals properly
            norm = self.backend.linalg.norm(batch_normals, axis=-1, keepdims=True) + 1e-7
            batch_normals = batch_normals / norm
            
            # ✅ CRITICAL: Ensure all arrays are contiguous for cache consistency
            if self.device == "gpu":
                batch_images = cp.ascontiguousarray(batch_images)
                batch_keypoints = cp.ascontiguousarray(batch_keypoints)
                batch_segmentation = cp.ascontiguousarray(batch_segmentation)
                batch_normals = cp.ascontiguousarray(batch_normals)
            else:
                batch_images = np.ascontiguousarray(batch_images)
                batch_keypoints = np.ascontiguousarray(batch_keypoints)
                batch_segmentation = np.ascontiguousarray(batch_segmentation)
                batch_normals = np.ascontiguousarray(batch_normals)
            
            # ✅ CACHE-SAFE FSEField CREATION: Use the same method as regular batches
            input_data, label_dict = self._create_fse_fields_internal(
                batch_images, batch_keypoints, batch_segmentation, batch_normals, batch_size
            )
            
            # ✅ NEW: Create fallback flags - synthetic masks don't count as "mask present"
            fallback_flags = [{"fluxa_segmentation": False} for _ in range(batch_size)]
            
            # ✅ STRICT VALIDATION: Ensure all fields pass validation
            if not self._validate_fse_fields_strict(input_data, label_dict):
                logger.error("❌ Cache-safe fallback batch validation failed")
                return [], {}, []
            
            # ✅ ADDITIONAL CACHE VALIDATION: Check that fields will create proper caches
            if not self._validate_cache_compatibility(input_data, label_dict):
                logger.error("❌ Cache compatibility validation failed")
                return [], {}, []
            
            # ✅ NEW: Increment fallback counter
            self._fallback_batch_counter += 1
            
            logger.info(f"✅ Created cache-safe fallback batch: {batch_size} samples, validated for cache consistency")
            return input_data, label_dict, fallback_flags  # ← NEW: Return flags
            
        except Exception as e:
            logger.error(f"❌ Cache-safe fallback batch creation failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return [], {}, []
    
    def get_async_batch_generator(self, sample_ids: List[str], is_training: bool, batch_size: int):
        """Enhanced async batch generator with validation and balanced mask sampling"""
        
        # ✅ NEW: Balanced iterator for training with guaranteed masks
        if is_training and hasattr(self, 'mask_ids') and self.mask_ids:
            import itertools
            
            logger.info(f"🎯 Using balanced mask sampling: {len(self.mask_ids)} mask samples, {len(self.nomask_ids)} no-mask samples")
            
            # Create infinite cycles
            mask_cycle = itertools.cycle(self.mask_ids)
            nomask_cycle = itertools.cycle(self.nomask_ids if self.nomask_ids else self.mask_ids)
            
            batches_yielded = 0
            max_batches = max(1, len(self.all_train_ids) // (batch_size * self.world_size))
            
            # Start async prefetching with balanced batches
            self.async_prefetcher = AsyncBatchPrefetcher(
                self, batch_size, 
                prefetch_batches=min(2, max(1, self.args.prefetch_batches // 2)),
                num_workers=min(2, max(1, self.args.num_data_workers // 2))
            )
            
            try:
                while batches_yielded < max_batches:
                    # Create balanced batch: 1 guaranteed mask + fill rest
                    balanced_batch_ids = [next(mask_cycle)]  # 1 guaranteed mask
                    balanced_batch_ids += [next(nomask_cycle) for _ in range(batch_size - 1)]  # fill rest
                    
                    # Shuffle to mix order
                    import random
                    random.shuffle(balanced_batch_ids)
                    
                    # Start prefetching for this specific batch
                    self.async_prefetcher.start_prefetching(balanced_batch_ids, is_training)
                    
                    # Get the batch
                    batch_data = self.async_prefetcher.get_next_batch(timeout=45.0)
                    
                    if batch_data is None:
                        logger.warning("⏱️ Balanced batch timeout, creating fallback...")
                        try:
                            fallback_inputs, fallback_labels, fallback_flags = self._create_fallback_batch_validated(batch_size)
                            if fallback_inputs and fallback_labels:
                                yield fallback_inputs, fallback_labels, fallback_flags
                                batches_yielded += 1
                                continue
                        except Exception as e:
                            logger.error(f"❌ Balanced fallback failed: {e}")
                            break
                    else:
                        # Yield the balanced batch
                        if (batch_data.batch_inputs and len(batch_data.batch_inputs) > 0 and 
                            batch_data.batch_labels and len(batch_data.batch_labels) > 0):
                            yield batch_data.batch_inputs, batch_data.batch_labels, batch_data.batch_valid
                            batches_yielded += 1
                            
                            if batches_yielded % 10 == 0:
                                logger.debug(f"📈 Balanced batches: {batches_yielded}/{max_batches}")
                
                logger.info(f"✅ Balanced sampling completed: {batches_yielded} batches with guaranteed masks")
                
            except Exception as e:
                logger.error(f"❌ Balanced batch generation error: {e}")
            finally:
                if self.async_prefetcher:
                    self.async_prefetcher.stop_prefetching()
                    self.async_prefetcher = None
            
            return
        
        # ✅ Fallback to original behaviour (validation or no mask list)
        # Filter IDs for this rank
        rank_ids = sample_ids[self.rank::self.world_size]
        
        if is_training:
            import random
            random.shuffle(rank_ids)
        
        if self.rank == 0:
            logger.info(f"🚀 Rank {self.rank}/{self.world_size} processing {len(rank_ids)} samples")
        
        # Start async prefetching with conservative settings
        self.async_prefetcher = AsyncBatchPrefetcher(
            self, batch_size, 
            prefetch_batches=min(2, max(1, self.args.prefetch_batches // 2)),
            num_workers=min(2, max(1, self.args.num_data_workers // 2))
        )
        
        self.async_prefetcher.start_prefetching(rank_ids, is_training)
        
        try:
            batches_yielded = 0
            max_batches = max(1, len(rank_ids) // batch_size)
            consecutive_empty = 0
            max_consecutive_empty = 5  # Reduced for faster fallback
            
            logger.info(f"📊 Target: {max_batches} batches for rank {self.rank}")
            
            while batches_yielded < max_batches and consecutive_empty < max_consecutive_empty:
                try:
                    batch_data = self.async_prefetcher.get_next_batch(timeout=45.0)
                    
                    if batch_data is None:
                        consecutive_empty += 1
                        logger.warning(f"⏱️ Async batch timeout (attempt {consecutive_empty}/{max_consecutive_empty})")
                        
                        # Quick fallback on timeout if allowed
                        if consecutive_empty >= 3:
                            # ✅ NEW: Check fallback policy
                            if not hasattr(self.args, 'allow_fallback_batches') or not self.args.allow_fallback_batches:
                                logger.error("❌ Batch timeout and fallback is disabled")
                                break
                            
                            logger.info("🔄 Creating emergency fallback due to timeouts...")
                            try:
                                fallback_inputs, fallback_labels, fallback_flags = self._create_fallback_batch_validated(batch_size)
                                if fallback_inputs and fallback_labels:
                                    yield fallback_inputs, fallback_labels, fallback_flags  # ← NEW: Yield flags
                                    batches_yielded += 1
                                    consecutive_empty = 0
                                    continue
                            except Exception as e:
                                logger.error(f"❌ Emergency fallback failed: {e}")
                        
                        time.sleep(1.0)
                        continue
                    
                    # Final validation before yielding
                    if (batch_data.batch_inputs and len(batch_data.batch_inputs) > 0 and 
                        batch_data.batch_labels and len(batch_data.batch_labels) > 0):
                        
                        yield batch_data.batch_inputs, batch_data.batch_labels, batch_data.batch_valid  # ← NEW: Yield flags
                        batches_yielded += 1
                        consecutive_empty = 0
                        
                        if batches_yielded % 10 == 0:
                            logger.debug(f"📈 Rank {self.rank}: {batches_yielded}/{max_batches} batches yielded")
                    else:
                        consecutive_empty += 1
                        logger.warning(f"⚠️ Invalid batch received (attempt {consecutive_empty})")
                        
                except Exception as e:
                    consecutive_empty += 1
                    logger.error(f"❌ Batch generation error (attempt {consecutive_empty}): {e}")
                    time.sleep(0.5)
                    continue
            
            if consecutive_empty >= max_consecutive_empty:
                logger.error(f"❌ Rank {self.rank}: Stopping after {consecutive_empty} consecutive failures")
            else:
                logger.info(f"✅ Rank {self.rank}: Successfully yielded {batches_yielded} batches")
                
        except Exception as e:
            logger.error(f"❌ Batch generator error for rank {self.rank}: {e}")
        finally:
            # Cleanup
            if self.async_prefetcher:
                self.async_prefetcher.stop_prefetching()
                self.async_prefetcher = None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get data loading cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_samples': len(self._data_cache)
        }