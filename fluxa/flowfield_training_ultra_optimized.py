# file: flowfield_training_ultra_optimized.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import sys
import time
import io
import json
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import cupy as cp
import h5py

# Import local FSE components
from flowfield_fluxa_model import ProductionFLUXA
from flowfield_core_optimized import get_memory_pool, get_default_dtype, FSEField, FieldType
from flowfield_async_data_loader import ProductionDataLoader
from metrics_fse import PhysicsInformedMetrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FLUXA_TRAINING")

def save_checkpoint(model, optimizer, epoch: int, step: int, loss: float, 
                   checkpoint_dir: str, rank: int = 0):
    """Save checkpoint locally with HDF5 format"""
    if rank != 0: return
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_e{epoch}_s{step}.h5")
        with h5py.File(path, 'w') as h5f:
            meta = h5f.create_group('metadata')
            meta.attrs['epoch'], meta.attrs['step'], meta.attrs['loss'] = epoch, step, loss
            model_group = h5f.create_group('model_parameters')
            _serialize_params_h5(model.parameters, model_group)
        logger.info(f"✅ Saved: {path}")
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")

def load_checkpoint(model, optimizer, load_path: str, device: str, rank: int = 0):
    """Load checkpoint locally from HDF5 format"""
    try:
        if not os.path.exists(load_path): return 0, 0, float('inf')
        with h5py.File(load_path, 'r') as h5f:
            meta = h5f['metadata']
            epoch, step, loss = int(meta.attrs['epoch']), int(meta.attrs['step']), float(meta.attrs['loss'])
            if 'model_parameters' in h5f:
                _deserialize_params_h5(model.parameters, h5f['model_parameters'], device)
        return epoch, step, loss
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        return 0, 0, float('inf')

def _serialize_params_h5(params, group):
    for k, v in params.items():
        if isinstance(v, dict): _serialize_params_h5(v, group.create_group(k))
        elif hasattr(v, 'data'):
            d = v.data
            if hasattr(d, 'get'): d = d.get()
            group.create_dataset(k, data=d, compression='gzip')

def _deserialize_params_h5(model_params, group, device):
    for k in group.keys():
        if isinstance(group[k], h5py.Group): _deserialize_params_h5(model_params[k], group[k], device)
        else:
            d = group[k][...]
            if device == "gpu": d = cp.asarray(d)
            model_params[k].data = d
