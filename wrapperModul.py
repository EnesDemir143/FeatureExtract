import torch.nn as nn
from .utils_feature import extract_feature_tensor

class wrapperModel(nn.Module):
    def __init__(self, backbone, head_mlp, use_processor=False):
        super().__init__()
        self.backbone = backbone
        self.head_mlp = head_mlp
        self.use_processor = use_processor 

    def forward(self, x):

        if self.use_processor:
            out = self.backbone(pixel_values=x, output_hidden_states=False, return_dict=True)
        else:
            out = self.backbone(x)

        feats = extract_feature_tensor(out)   
        return self.head_mlp(feats)          
