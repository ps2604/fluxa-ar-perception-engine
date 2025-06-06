# file: metrics_fse.py
# COMPREHENSIVE FSE METRICS: Restored per-head metrics (MAE, mIoU, FSE coherence, physics accuracy) + SEGMENTATION FIX
# UPDATES: Added all missing metrics computation for FlowField FSE architecture, added segmentation mask validation
# FIXES: CuPy gradient compatibility and proper spatial gradient computation, S-2: Added empty mask guard

import numpy as np
import cupy as cp
from typing import Dict, Any, Optional, Union
import logging

from flowfield_core_optimized import FSEField, FieldType

logger = logging.getLogger(__name__)

def _spatial_grad(x):
    """
    Forward-difference gradients (no pad) for a 4-D tensor [B,H,W,C].
    Returns dx, dy with the same shape.
    Works for NumPy or CuPy arrays.
    """
    lib = cp if isinstance(x, cp.ndarray) else np
    dx = lib.zeros_like(x)
    dy = lib.zeros_like(x)

    # width-direction (W axis)
    dx[..., :-1, :] = x[..., 1:, :]  - x[..., :-1, :]

    # height-direction (H axis)
    dy[:, :-1, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]

    return dx, dy



class FSEMetricsComputer:
    """Comprehensive FSE metrics computer for all modalities"""
    
    def __init__(self, device: str = "gpu"):
        self.device = device
        self.backend = cp if device == "gpu" else np
        
    def compute_all_metrics(self, predictions: Dict[str, FSEField], targets: Dict[str, FSEField]) -> Dict[str, float]:
        """Compute all comprehensive FSE metrics"""
        metrics = {}
        
        try:
            # Keypoints metrics
            if 'fluxa_keypoints' in predictions and 'fluxa_keypoints' in targets:
                kp_metrics = self.compute_keypoints_metrics(
                    predictions['fluxa_keypoints'], targets['fluxa_keypoints']
                )
                metrics.update(kp_metrics)
            
            # Segmentation metrics  
            if 'fluxa_segmentation' in predictions and 'fluxa_segmentation' in targets:
                seg_metrics = self.compute_segmentation_metrics(
                    predictions['fluxa_segmentation'], targets['fluxa_segmentation']
                )
                metrics.update(seg_metrics)
            
            # Surface normals metrics
            if 'fluxa_surface_normals' in predictions and 'fluxa_surface_normals' in targets:
                sn_metrics = self.compute_surface_normals_metrics(
                    predictions['fluxa_surface_normals'], targets['fluxa_surface_normals']
                )
                metrics.update(sn_metrics)
            
            # Environment lighting metrics
            if 'fluxa_environment_lighting' in predictions and 'fluxa_environment_lighting' in targets:
                env_metrics = self.compute_environment_lighting_metrics(
                    predictions['fluxa_environment_lighting'], targets['fluxa_environment_lighting']
                )
                metrics.update(env_metrics)
            
            # Global FSE metrics
            global_metrics = self.compute_global_fse_metrics(predictions, targets)
            metrics.update(global_metrics)
            
        except Exception as e:
            logger.debug(f"Metrics computation failed: {e}")
            
        return metrics
    
    def compute_keypoints_metrics(self, pred: FSEField, target: FSEField) -> Dict[str, float]:
        """Compute keypoints-specific metrics"""
        metrics = {}
        
        try:
            # Ensure same device
            if pred.device != target.device:
                target = target.to_device(pred.device)
            
            # MAE
            mae = float(self.backend.mean(self.backend.abs(pred.data - target.data)))
            metrics['fluxa_keypoints_mae'] = mae
            
            # FSE Field Coherence
            coherence = self.compute_fse_coherence(pred, target)
            metrics['fluxa_keypoints_fse_coherence'] = coherence
            
            # Keypoint detection accuracy (for non-zero keypoints)
            kp_threshold = 0.1
            pred_binary = (pred.data > kp_threshold).astype(self.backend.float32)
            target_binary = (target.data > kp_threshold).astype(self.backend.float32)
            
            intersection = self.backend.sum(pred_binary * target_binary)
            union = self.backend.sum(pred_binary + target_binary - pred_binary * target_binary)
            
            if union > 0:
                kp_accuracy = float(intersection / union)
            else:
                kp_accuracy = 1.0
            
            metrics['fluxa_keypoints_detection_accuracy'] = kp_accuracy
            
        except Exception as e:
            logger.debug(f"Keypoints metrics failed: {e}")
            
        return metrics
    
    def compute_segmentation_metrics(self, pred: FSEField, target: FSEField) -> Dict[str, float]:
        """Accuracy, mIoU and Dice for binary segmentation masks with empty mask guard."""
        THRESH = 0.10                      # lower than default 0.5 to catch faint masks
        metrics = {}
        try:
            if pred.device != target.device:
                target = target.to_device(pred.device)

            # Rescale 0-255 masks if needed
            tgt = target.data
            if tgt.max() > 1.5:
                tgt = tgt / 255.0

            # ✅ S-2 NEW: Early exit guard for empty masks
            tgt_bin = (tgt >= THRESH).astype(self.backend.float32)
            if tgt_bin.sum() < 1:  # mask is empty → skip metric
                metrics['fluxa_segmentation_accuracy'] = 0.0
                metrics['fluxa_segmentation_miou'] = 0.0
                metrics['fluxa_segmentation_dice'] = 0.0
                return metrics

            pred_bin = (pred.data >= THRESH).astype(self.backend.float32)

            correct = self.backend.sum(pred_bin == tgt_bin)
            metrics["fluxa_segmentation_accuracy"] = float(correct / pred_bin.size)

            inter  = self.backend.sum(pred_bin * tgt_bin)
            union  = self.backend.sum(pred_bin + tgt_bin - pred_bin * tgt_bin)
            metrics["fluxa_segmentation_miou"] = float(inter / union) if union else 1.0

            denom = self.backend.sum(pred_bin) + self.backend.sum(tgt_bin)
            metrics["fluxa_segmentation_dice"] = float(2 * inter / denom) if denom else 1.0

        except Exception as e:
            logger.debug(f"Segmentation metrics failed: {e}")
        return metrics
    
    def compute_surface_normals_metrics(self, pred: FSEField, target: FSEField) -> Dict[str, float]:
        metrics = {}
        try:
            if pred.device != target.device:
                target = target.to_device(pred.device)

            mae = float(self.backend.mean(self.backend.abs(pred.data - target.data)))
            metrics['fluxa_surface_normals_mae'] = mae

            # angular error
            pred_norm = pred.data / (self.backend.linalg.norm(pred.data, axis=-1, keepdims=True) + 1e-7)
            tgt_norm  = target.data / (self.backend.linalg.norm(target.data, axis=-1, keepdims=True) + 1e-7)
            dot = self.backend.clip(self.backend.sum(pred_norm * tgt_norm, axis=-1), -1.0, 1.0)
            metrics['fluxa_surface_normals_angular_error'] = float(self.backend.mean(self.backend.arccos(self.backend.abs(dot))))

            # physics accuracy – call the helper you already defined
            metrics['fluxa_surface_normals_physics_accuracy'] = sn_physics_accuracy(pred, target)

            # coherence
            metrics['fluxa_surface_normals_fse_coherence'] = self.compute_fse_coherence(pred, target)
        except Exception as e:
            logger.debug(f"Normals metrics failed: {e}")
        return metrics
    
    def compute_environment_lighting_metrics(self, pred: FSEField, target: FSEField) -> Dict[str, float]:
        """Compute environment lighting-specific metrics"""
        metrics = {}
        
        try:
            # Ensure same device
            if pred.device != target.device:
                target = target.to_device(pred.device)
            
            # MAE
            mae = float(self.backend.mean(self.backend.abs(pred.data - target.data)))
            metrics['fluxa_environment_lighting_mae'] = mae
            
            # MSE
            mse = float(self.backend.mean((pred.data - target.data) ** 2))
            metrics['fluxa_environment_lighting_mse'] = mse
            
            # Correlation coefficient
            pred_flat = pred.data.flatten()
            target_flat = target.data.flatten()
            
            if len(pred_flat) > 1:
                correlation = float(self.backend.corrcoef(pred_flat, target_flat)[0, 1])
                if not self.backend.isnan(correlation):
                    metrics['fluxa_environment_lighting_correlation'] = correlation
            
        except Exception as e:
            logger.debug(f"Environment lighting metrics failed: {e}")
            
        return metrics
    
    def compute_fse_coherence(self, pred: FSEField, target: FSEField) -> float:
        """Spatial-gradient coherence between predicted & GT fields."""
        try:
            if pred.device != target.device:
                target = target.to_device(pred.device)

            dx_p, dy_p = _spatial_grad(pred.data)
            dx_t, dy_t = _spatial_grad(target.data)

            mag_p = self.backend.sqrt(dx_p**2 + dy_p**2)
            mag_t = self.backend.sqrt(dx_t**2 + dy_t**2)

            grad_diff = self.backend.mean(self.backend.abs(mag_p - mag_t))
            return float(1.0 / (1.0 + grad_diff))  # higher = better
        except Exception as e:
            logger.debug(f"Coherence failed: {e}")
            return 0.5
    def compute_global_fse_metrics(self, predictions: Dict[str, FSEField], targets: Dict[str, FSEField]) -> Dict[str, float]:
        """Compute global FSE system metrics"""
        metrics = {}
        
        try:
            # Overall field coherence across all modalities
            all_coherences = []
            for modality in predictions.keys():
                if modality in targets:
                    coherence = self.compute_fse_coherence(predictions[modality], targets[modality])
                    all_coherences.append(coherence)
            
            if all_coherences:
                global_coherence = float(self.backend.mean(self.backend.array(all_coherences)))
                metrics['global_field_coherence'] = global_coherence
            
            # System prediction consistency
            if len(predictions) > 1:
                field_consistency = self.compute_cross_field_consistency(predictions)
                metrics['global_field_consistency'] = field_consistency
            
        except Exception as e:
            logger.debug(f"Global FSE metrics failed: {e}")
            
        return metrics
    
    def compute_cross_field_consistency(self, predictions: Dict[str, FSEField]) -> float:
        """Compute consistency across different field predictions"""
        try:
            # Simple consistency metric based on field activation patterns
            activations = []
            for field in predictions.values():
                if len(field.shape) >= 3:  # Spatial fields
                    activation = float(self.backend.mean(self.backend.abs(field.data)))
                    activations.append(activation)
            
            if len(activations) > 1:
                activation_std = float(self.backend.std(self.backend.array(activations)))
                activation_mean = float(self.backend.mean(self.backend.array(activations)))
                consistency = 1.0 / (1.0 + activation_std / (activation_mean + 1e-7))
                return consistency
            
            return 1.0
            
        except Exception as e:
            logger.debug(f"Cross-field consistency failed: {e}")
            return 0.5

# Backward compatibility functions for direct usage
def compute_fse_metrics(predictions: Dict[str, FSEField], targets: Dict[str, FSEField], 
                       device: str = "gpu") -> Dict[str, float]:
    """Compute comprehensive FSE metrics - backward compatibility function"""
    computer = FSEMetricsComputer(device)
    return computer.compute_all_metrics(predictions, targets)

# Individual metric functions for specific use cases
def kp_mae(pred: FSEField, target: FSEField) -> float:
    """Keypoint MAE"""
    backend = cp if pred.device == "gpu" else np
    if pred.device != target.device:
        target = target.to_device(pred.device)
    return float(backend.mean(backend.abs(pred.data - target.data)))

def kp_coherence(pred: FSEField, target: FSEField) -> float:
    """Keypoint FSE coherence"""
    computer = FSEMetricsComputer(pred.device)
    return computer.compute_fse_coherence(pred, target)

def seg_accuracy(pred: FSEField, target: FSEField) -> float:
    """Segmentation accuracy"""
    backend = cp if pred.device == "gpu" else np
    if pred.device != target.device:
        target = target.to_device(pred.device)
    
    pred_binary = (pred.data > 0.5).astype(backend.float32)
    target_binary = (target.data > 0.5).astype(backend.float32)
    
    correct = backend.sum(pred_binary == target_binary)
    total = pred_binary.size
    return float(correct / total)

def seg_miou(pred: FSEField, target: FSEField) -> float:
    """Segmentation mIoU"""
    backend = cp if pred.device == "gpu" else np
    if pred.device != target.device:
        target = target.to_device(pred.device)
    
    pred_binary = (pred.data > 0.5).astype(backend.float32)
    target_binary = (target.data > 0.5).astype(backend.float32)
    
    intersection = backend.sum(pred_binary * target_binary)
    union = backend.sum(pred_binary + target_binary - pred_binary * target_binary)
    
    if union > 0:
        return float(intersection / union)
    return 1.0

def sn_mae(pred: FSEField, target: FSEField) -> float:
    """Surface normals MAE"""
    backend = cp if pred.device == "gpu" else np
    if pred.device != target.device:
        target = target.to_device(pred.device)
    return float(backend.mean(backend.abs(pred.data - target.data)))

def sn_physics_accuracy(pred: FSEField, target: FSEField) -> float:
    """
    Physics-grounded accuracy for surface normals
    (matches the original TensorFlow implementation).
    Returns a scalar in [0, 1]; higher = better.
    """
    backend = cp if pred.device == "gpu" else np
    eps = 1e-8

    # 1) Unit-length term ────────────────────────────────────────────────
    norm_mag   = backend.linalg.norm(pred.data, axis=-1) + eps
    unit_dev   = backend.abs(norm_mag - 1.0)
    unit_score = backend.exp(-4.0 * unit_dev)                    # ∈ (0,1]

    # 2) Smoothness term (finite-difference gradients) ──────────────────
    dx, dy = _spatial_grad(pred.data)
    grad_mag_x = backend.sqrt((dx ** 2).sum(axis=-1) + eps)
    grad_mag_y = backend.sqrt((dy ** 2).sum(axis=-1) + eps)

    tgt_grad = 0.12                                              # desired avg gradient
    grad_score_x = backend.exp(-10.0 * backend.abs(grad_mag_x.mean(axis=(1, 2)) - tgt_grad))
    grad_score_y = backend.exp(-10.0 * backend.abs(grad_mag_y.mean(axis=(1, 2)) - tgt_grad))
    grad_score   = 0.5 * (grad_score_x + grad_score_y)

    # 3) Distribution term (global normal orientation) ─────────────────
    mean_normal = pred.data.mean(axis=(1, 2))
    dist_score  = backend.exp(-6.0 * (backend.linalg.norm(mean_normal, axis=-1) - 0.35) ** 2)

    # Weighted combination (same weights as TF implementation) ─────────
    physics = 0.4 * unit_score.mean(axis=(1, 2)) + 0.4 * grad_score + 0.2 * dist_score
    return float(physics.mean())

def sn_coherence(pred: FSEField, target: FSEField) -> float:
    """Surface normals FSE coherence"""
    computer = FSEMetricsComputer(pred.device)
    return computer.compute_fse_coherence(pred, target)

def env_mae(pred: FSEField, target: FSEField) -> float:
    """Environment lighting MAE"""
    backend = cp if pred.device == "gpu" else np
    if pred.device != target.device:
        target = target.to_device(pred.device)
    return float(backend.mean(backend.abs(pred.data - target.data)))