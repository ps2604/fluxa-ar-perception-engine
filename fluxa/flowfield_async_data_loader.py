# file: flowfield_async_data_loader.py (Cleaned Version)
# Removed all GCP/GCS dependencies for local research archive.

import os
import sys
import time
import json
import random
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import cupy as cp

logger = logging.getLogger("FLUXA_DATA_LOADER")

class ProductionDataLoader:
    """[LOCAL ARCHIVE VERSION] Cleaned local-only data loader."""
    def __init__(self, data_dir: str, batch_size: int, device: str = "gpu"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.backend = cp if device == "gpu" else np
        
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            
    def load_sample(self, sample_id: str):
        path = os.path.join(self.data_dir, f"{sample_id}.json")
        with open(path, 'r') as f:
            return json.load(f)
