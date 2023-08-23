import torch.nn as nn

class ROI_Pool(nn.Module):
    
    def __init__(self, size):
        super(ROI_Pool, self).__init__()
        assert len(size) == 2
        pool_func = nn.AdaptiveMaxPool2d
        
        self.roi_pool = pool_func(size)
        
    
    def forward(self, feature_maps):
        assert feature_maps.dim() == 4
        return  self.roi_pool(feature_maps)