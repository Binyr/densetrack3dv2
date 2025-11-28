import torch
from einops import rearrange


REPO_DIR = "/mnt/shared-storage-user/binyanrui/Projects/dinov3"
DINO_DIR = "/mnt/shared-storage-user/si-data/DINOv3/"
pretrained_model_name_or_path = f"{DINO_DIR}dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

class DINOv3_Encoder(torch.nn.Module):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name = 'dinov2_vitl16',
        freeze = True,
        antialias=True,
        device="cuda",
        size = 1024,
        dtype=torch.bfloat16
    ):
        super(DINOv3_Encoder, self).__init__()
        
        self.model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=pretrained_model_name_or_path)
        self.model.eval().to(device).to(dtype)
        self.device = device
        self.antialias = antialias
        self.dtype = dtype

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)
        self.size = size
        if freeze:
            self.freeze()


    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # @torch.no_grad()
    def encoder(self, x, size=None):
        '''
        x: [b h w c], range from (-1, 1), rbg
        '''
        ori_dtype = x.dtype
        self.to(self.dtype)
        x = x.to(self.dtype)
        x = self.preprocess(x, size).to(self.device, self.dtype)

        b, c, h, w = x.shape
        patch_h, patch_w = h // 16, w // 16

        embeddings = self.model.forward_features(x)['x_norm_patchtokens']
        embeddings = rearrange(embeddings, 'b (h w) c -> b h w c', h = patch_h, w = patch_w)

        embeddings = embeddings.to(ori_dtype)
        return  rearrange(embeddings, 'b h w c -> b c h w')

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