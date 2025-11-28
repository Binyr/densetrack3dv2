from pathlib import Path
f_path = Path(__file__).absolute()
import sys
sys.path.insert(0, "/mnt/shared-storage-user/binyanrui/Projects/sam2")

import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor



class Sam2Encoder(torch.nn.Module):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    def __init__(self, 
                 model_path="facebook/sam2-hiera-large",
                 device="cuda",
                 antialias=True,
                 size = 448
                ):
        super(Sam2Encoder, self).__init__()
        self.model = SAM2VideoPredictor.from_pretrained(model_path)
        # self.model.use_high_res_features_in_sam = False

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)

        self.size = size
        self.device = device
        self.antialias = antialias

    @torch.no_grad()
    def encoder(self, x, size=None):
        '''
        x: [b h w c], range from (-1, 1), rbg
        '''

        x = self.preprocess(x, size).to(self.device, self.dtype)

        backbone_out = self.model.forward_image(x)["backbone_fpn"][0]

        return backbone_out
    
    def preprocess(self, x, size=None):
        ''' x
        '''
        # normalize to [0,1],
        if size is None:
            size = (self.size, self.size)
        x = torch.nn.functional.interpolate(
            x,
            size=size,
            mode='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )

        x = (x + 1.0) / 2.0
        # renormalize according to dino
        mean = self.mean.view(1, 3, 1, 1).to(x.device)
        std = self.std.view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        return x
    
    def to(self, device, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.model.to(device, dtype)
            self.mean.to(device, dtype)
            self.std.to(device, dtype)
        else:
            self.model.to(device)
            self.mean.to(device)
            self.std.to(device)
        return self

    def __call__(self, x, **kwargs):
        return self.encoder(x, **kwargs)
