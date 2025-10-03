"""
Configuration file for DEAL project.
Centralizes path management using environment variables.
"""
import os
from pathlib import Path

# Base paths - users should modify these via environment variables
# Default assumes datasets are in user's home directory under resources/datasets
DEFAULT_DATASET_ROOT = os.environ.get('DATASET_ROOT', os.path.expanduser('~/resources/datasets'))
DEFAULT_CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_ROOT', f'{DEFAULT_DATASET_ROOT}/cityscapes')

# CUDA settings
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()

class Config:
    """Configuration class for DEAL project paths and settings."""
    
    def __init__(self):
        self.dataset_root = Path(DEFAULT_DATASET_ROOT).resolve()
        self.cityscapes_root = Path(DEFAULT_CITYSCAPES_ROOT).resolve()
        self.cuda_home = CUDA_HOME
        self.project_root = PROJECT_ROOT
        
        # Cityscapes specific paths
        self.cityscapes_leftimg_root = self.cityscapes_root / "leftImg8bit"
        self.cityscapes_gtfine_root = self.cityscapes_root / "gtFine-relabeled"
        
        # Regional CSV path
        self.cityscapes_region_csv = self.project_root / "datasets" / "cityscapes" / "region_list_seed20_budget480_initialization.csv"
    
    def get_cityscapes_paths(self, split='train'):
        """Get Cityscapes image and target paths for a given split."""
        img_paths_file = self.cityscapes_root / f"{split}_img_paths.txt"
        target_paths_file = self.cityscapes_root / f"{split}_target_paths.txt"
        return str(img_paths_file), str(target_paths_file)
    
    def convert_legacy_path(self, path):
        """Convert legacy hard-coded paths to current configuration."""
        path = str(path)
        
        # Convert various legacy path formats
        legacy_mappings = [
            ('/home/ammiellewb/resources/datasets/cityscapes', str(self.cityscapes_root)),
            ('/mnt/resources/datasets/cityscapes', str(self.cityscapes_root)),
            ('/nfs/xs/Datasets/Segment/Cityscapes', str(self.cityscapes_root)),
            ('/home/ammiellewb/DEAL', str(self.project_root)),
            ('/nfs/xs/codes/DEAL', str(self.project_root)),
        ]
        
        for old_path, new_path in legacy_mappings:
            if path.startswith(old_path):
                return path.replace(old_path, new_path)
        
        return path

# Global config instance
config = Config()