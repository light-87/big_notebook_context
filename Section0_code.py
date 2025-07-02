# ============================================================================
# SECTION 0: SETUP & CONFIGURATION
# ============================================================================

# Standard library imports
import os
import gc
import multiprocessing as mp
import sys
import json
import time
import logging
import warnings
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle

# Data manipulation
import numpy as np
import pandas as pd
import datatable as dt
from datatable import f, by

# Machine Learning
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Deep Learning
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image

# Progress tracking
from tqdm.auto import tqdm
import progressbar

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Global Configuration
# ============================================================================

# Core Parameters
WINDOW_SIZE = 20                    # Sequence window around phosphorylation site
RANDOM_SEED = 42                    # For reproducibility
EXPERIMENT_NAME = "exp_3"           # Experiment identifier
BASE_DIR = f"results/{EXPERIMENT_NAME}"
MAX_SEQUENCE_LENGTH = 5000          # Filter long sequences
BALANCE_CLASSES = True              # 1:1 positive:negative ratio
USE_DATATABLE = True                # Use datatable for speed optimization
BATCH_SIZE = 32                     # For transformer training
GRADIENT_ACCUMULATION_STEPS = 2     # Memory optimization
USE_MIXED_PRECISION = True          # For transformer efficiency

# Set all random seeds for reproducibility
def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_all_seeds(RANDOM_SEED)

# ============================================================================
# Progress Tracking System
# ============================================================================

class ProgressTracker:
    """Comprehensive progress tracking with checkpoint management"""
    
    def __init__(self, exp_dir: str, auto_cleanup: bool = True):
        self.exp_dir = exp_dir
        self.auto_cleanup = auto_cleanup
        self.progress_file = os.path.join(exp_dir, 'progress_tracker.json')
        self.start_time = datetime.now()
        
        # Create directory structure
        self._create_directories()
        
        # Load or initialize progress
        self.progress = self._load_progress()
        
        # Memory monitoring
        self.memory_threshold = 0.8  # 80% memory usage triggers cleanup
        
    def _create_directories(self):
        """Create all required directories"""
        directories = [
            self.exp_dir,
            os.path.join(self.exp_dir, 'checkpoints'),
            os.path.join(self.exp_dir, 'checkpoints/data_preprocessing'),
            os.path.join(self.exp_dir, 'checkpoints/feature_extraction'),
            os.path.join(self.exp_dir, 'checkpoints/ml_models'),
            os.path.join(self.exp_dir, 'checkpoints/transformers'),
            os.path.join(self.exp_dir, 'checkpoints/ensemble'),
            os.path.join(self.exp_dir, 'ml_models'),
            os.path.join(self.exp_dir, 'transformers'),
            os.path.join(self.exp_dir, 'ensemble'),
            os.path.join(self.exp_dir, 'final_report'),
            os.path.join(self.exp_dir, 'logs'),
            os.path.join(self.exp_dir, 'plots'),
            os.path.join(self.exp_dir, 'plots/data_exploration'),
            os.path.join(self.exp_dir, 'plots/feature_analysis'),
            os.path.join(self.exp_dir, 'plots/ml_models'),
            os.path.join(self.exp_dir, 'plots/transformers'),
            os.path.join(self.exp_dir, 'plots/ensemble'),
            os.path.join(self.exp_dir, 'plots/error_analysis'),
            os.path.join(self.exp_dir, 'plots/final_evaluation'),
            os.path.join(self.exp_dir, 'plots/final_report'),
            os.path.join(self.exp_dir, 'tables'),
            os.path.join(self.exp_dir, 'models')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_progress(self) -> Dict:
        """Load progress from file if exists"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'experiment_start': self.start_time.isoformat(),
                'completed_steps': {},
                'checkpoints': {},
                'metadata': {
                    'experiment_name': EXPERIMENT_NAME,
                    'random_seed': RANDOM_SEED,
                    'window_size': WINDOW_SIZE
                }
            }
    
    def _save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def mark_completed(self, step_name: str, metadata: Dict = None, checkpoint_data: Any = None):
        """Mark a step as completed and optionally save checkpoint"""
        completion_time = datetime.now()
        self.progress['completed_steps'][step_name] = {
            'completed_at': completion_time.isoformat(),
            'duration_seconds': (completion_time - self.start_time).total_seconds(),
            'metadata': metadata or {}
        }
        
        if checkpoint_data is not None:
            checkpoint_path = os.path.join(
                self.exp_dir, 'checkpoints', f'{step_name.replace(" ", "_").lower()}.pkl'
            )
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=4)
            self.progress['checkpoints'][step_name] = checkpoint_path
        
        self._save_progress()
        
        # Check memory and cleanup if needed
        if self.auto_cleanup:
            self._check_memory_usage()
    
    def is_completed(self, step_name: str) -> bool:
        """Check if a step is already completed"""
        return step_name in self.progress['completed_steps']
    
    def resume_from_checkpoint(self, step_name: str) -> Any:
        """Resume from a checkpoint if exists"""
        if step_name in self.progress['checkpoints']:
            checkpoint_path = self.progress['checkpoints'][step_name]
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def get_progress_summary(self) -> Dict:
        """Get summary of progress"""
        total_steps = 10  # Total number of major sections
        completed_steps = len(self.progress['completed_steps'])
        
        return {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'percentage': (completed_steps / total_steps) * 100,
            'elapsed_time': str(datetime.now() - self.start_time),
            'completed': list(self.progress['completed_steps'].keys())
        }
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed"""
        memory = self.get_memory_usage()
        if memory['percent'] > self.memory_threshold * 100:
            self.trigger_cleanup()
    
    def trigger_cleanup(self):
        """Trigger memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def force_retrain(self, step_name: str):
        """Force retrain by removing a completed step"""
        if step_name in self.progress['completed_steps']:
            del self.progress['completed_steps'][step_name]
        if step_name in self.progress['checkpoints']:
            checkpoint_path = self.progress['checkpoints'][step_name]
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            del self.progress['checkpoints'][step_name]
        self._save_progress()
    
    def export_progress_report(self) -> str:
        """Export detailed progress report"""
        report = f"""
Phosphorylation Prediction Experiment Progress Report
=====================================================
Experiment: {EXPERIMENT_NAME}
Started: {self.progress['experiment_start']}
Current Time: {datetime.now().isoformat()}
Elapsed: {datetime.now() - self.start_time}

Progress Summary:
-----------------
"""
        summary = self.get_progress_summary()
        report += f"Completed: {summary['completed_steps']}/{summary['total_steps']} steps ({summary['percentage']:.1f}%)\n\n"
        
        report += "Completed Steps:\n"
        for step, info in self.progress['completed_steps'].items():
            report += f"- {step}: {info['completed_at']} (Duration: {info['duration_seconds']:.1f}s)\n"
        
        report += f"\nMemory Usage:\n"
        memory = self.get_memory_usage()
        report += f"- RSS: {memory['rss_mb']:.1f} MB\n"
        report += f"- VMS: {memory['vms_mb']:.1f} MB\n"
        report += f"- Percent: {memory['percent']:.1f}%\n"
        
        return report

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str):
    """Setup comprehensive logging"""
    log_file = os.path.join(log_dir, 'experiment.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"Phosphorylation Prediction Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("="*80)
    
    return logger


# ============================================================================
# Environment Information
# ============================================================================

def log_environment_info(logger):
    """Log complete environment information"""
    logger.info("\nEnvironment Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA available: No (CPU mode)")
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total RAM: {memory.total / 1e9:.1f} GB")
        logger.info(f"Available RAM: {memory.available / 1e9:.1f} GB")
    except ImportError:
        logger.info("psutil not available for memory information")


# ============================================================================
# Configuration Export
# ============================================================================

def export_configuration(exp_dir: str):
    """Export complete experiment configuration"""
    config = {
        'experiment': {
            'name': EXPERIMENT_NAME,
            'base_dir': BASE_DIR,
            'created_at': datetime.now().isoformat()
        },
        'data': {
            'window_size': WINDOW_SIZE,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'balance_classes': BALANCE_CLASSES,
            'use_datatable': USE_DATATABLE
        },
        'training': {
            'random_seed': RANDOM_SEED,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'use_mixed_precision': USE_MIXED_PRECISION
        },
        'environment': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    config_file = os.path.join(exp_dir, 'experiment_config.yaml')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return config


# ============================================================================
# Initialize Everything
# ============================================================================

print("Initializing Phosphorylation Prediction Experiment...")
print(f"Experiment Name: {EXPERIMENT_NAME}")
print(f"Base Directory: {BASE_DIR}")

# Initialize progress tracker
progress_tracker = ProgressTracker(BASE_DIR)

# Setup logging
logger = setup_logging(os.path.join(BASE_DIR, 'logs'))

# Log environment information
log_environment_info(logger)

# Export configuration
config = export_configuration(BASE_DIR)
logger.info(f"Configuration exported to: {os.path.join(BASE_DIR, 'experiment_config.yaml')}")

# Display progress summary
summary = progress_tracker.get_progress_summary()
print(f"\nProgress: {summary['completed_steps']}/{summary['total_steps']} steps completed ({summary['percentage']:.1f}%)")
if summary['completed_steps'] > 0:
    print("Completed steps:", ", ".join(summary['completed']))

print("\nSetup completed successfully!")
print("="*80)