"""
Utilities package for hierarchical medical image classification
"""

# Model architectures
from .base_model import Base3DCNN, Enhanced3DCNN, ResidualBlock3D
from .cnn_3d_models import (
    resnet18_3d, resnet34_3d, resnet50_3d,
    densenet121_3d, efficientnet3d_b0, get_3d_model
)
from .coarse_classifier import CoarseAnatomicalClassifier
from .fine_classifier import FinePathologyClassifier, RegionSpecificPathologyNetwork
from .subtype_classifier import SubtypeClassifier, HierarchicalSubtypeNetwork
from .hierarchical_model import HierarchicalClassificationModel

# Data utilities
from .data_loader import (
    get_medmnist_dataloaders,
    create_hierarchical_dataset,
    HierarchicalMedMNISTDataset,
    REGION_DATASET_MAPPING,
    DATASET_TO_REGION
)

# Training utilities
from .trainer import Trainer, HierarchicalTrainer

# Evaluation utilities
from .metrics import (
    compute_metrics,
    compute_hierarchical_metrics,
    evaluate_model,
    hierarchical_consistency_score
)

# Visualization utilities
from .visualization import (
    plot_training_history,
    visualize_3d_sample,
    plot_confusion_matrix,
    plot_hierarchical_results,
    plot_metrics_comparison
)

# Model management utilities
from .model_utils import (
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    print_model_summary,
    get_model_size,
    freeze_layers,
    unfreeze_all_layers,
    export_model_to_onnx,
    compare_models
)

__all__ = [
    # Models
    'Base3DCNN',
    'Enhanced3DCNN',
    'ResidualBlock3D',
    'CoarseAnatomicalClassifier',
    'MultiScaleCoarseClassifier',
    'FinePathologyClassifier',
    'RegionSpecificPathologyNetwork',
    'AttentionFineClassifier',
    'SubtypeClassifier',
    'HierarchicalSubtypeNetwork',
    'HierarchicalClassificationModel',
    
    # Data
    'get_medmnist_dataloaders',
    'create_hierarchical_dataset',
    'HierarchicalMedMNISTDataset',
    'REGION_DATASET_MAPPING',
    'DATASET_TO_REGION',
    
    # Training
    'Trainer',
    'HierarchicalTrainer',
    
    # Metrics
    'compute_metrics',
    'compute_hierarchical_metrics',
    'evaluate_model',
    'hierarchical_consistency_score',
    
    # Visualization
    'plot_training_history',
    'visualize_3d_sample',
    'plot_confusion_matrix',
    'plot_hierarchical_results',
    'plot_metrics_comparison',
    
    # Model utilities
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    'print_model_summary',
    'get_model_size',
    'freeze_layers',
    'unfreeze_all_layers',
    'export_model_to_onnx',
    'compare_models',
]
