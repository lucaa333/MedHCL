"""
Stage 1: Coarse-level anatomical region classifier
Classifies images into anatomical regions (brain, abdomen, chest)
"""

import torch
import torch.nn as nn
from .base_model import Base3DCNN, Enhanced3DCNN


class CoarseAnatomicalClassifier(nn.Module):
    """
    Stage 1 classifier for anatomical region localization.
    
    This classifier determines which anatomical region (brain, abdomen, chest)
    a medical image belongs to, mimicking the first step of radiological analysis.
    
    Args:
        architecture (str): Model architecture to use ('base' or 'enhanced')
        num_regions (int): Number of anatomical regions (default: 3)
        dropout_rate (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, architecture='base', num_regions=3, dropout_rate=0.3, region_names=None):
        super(CoarseAnatomicalClassifier, self).__init__()
        
        self.num_regions = num_regions
        self.architecture = architecture
        
        if architecture == 'enhanced':
            self.model = Enhanced3DCNN(
                in_channels=1,
                num_classes=num_regions,
                dropout_rate=dropout_rate
            )
        else:
            self.model = Base3DCNN(
                in_channels=1,
                num_classes=num_regions,
                dropout_rate=dropout_rate
            )
        
        # Region mapping - use provided mapping or default
        if region_names is None:
            self.region_names = {
                0: 'brain',
                1: 'abdomen', 
                2: 'chest'
            }
        else:
            self.region_names = region_names
    
    def forward(self, x):
        """
        Forward pass to classify anatomical region.
        
        Args:
            x (torch.Tensor): Input 3D medical images
        
        Returns:
            torch.Tensor: Logits for each anatomical region
        """
        return self.model(x)
    
    def predict_region(self, x):
        """
        Predict anatomical region with confidence scores.
        
        Args:
            x (torch.Tensor): Input 3D medical images
        
        Returns:
            tuple: (region_indices, region_names, confidence_scores)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, region_indices = torch.max(probabilities, dim=1)
        
        region_names = [self.region_names[idx.item()] for idx in region_indices]
        
        return region_indices, region_names, confidence_scores
    
    def get_region_name(self, region_idx):
        """Get region name from index."""
        return self.region_names.get(region_idx, 'unknown')


