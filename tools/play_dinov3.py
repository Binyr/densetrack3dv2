import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from einops import rearrange

from PIL import Image
import numpy as np
import pdb

REPO_DIR = "/mnt/shared-storage-user/binyanrui/Projects/dinov3"
DINO_DIR = "/mnt/shared-storage-user/si-data/DINOv3/"
pretrained_model_name_or_path = f"{DINO_DIR}dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=pretrained_model_name_or_path)

def vis_dino(feats, image):
    _, _, H, W = image.shape
    _, _, FH, FW = feats.shape
    selected_point = [H // 2, W // 2]
    feats_up = torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear")
    feats_up_vis = feats_up[:, :3]
    feats_up_vis = (feats_up_vis - feats_up_vis.min()) / (feats_up_vis.max() - feats_up_vis.min())
    image = image * 0.5 + 0.5

    # similarity map
    # s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up)
    s_map = torch.nn.functional.cosine_similarity(feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1), feats_up, dim=1)
    s_map = s_map.unsqueeze(1).repeat(1,3,1,1)
    pdb.set_trace()
    vis = torch.cat([image, feats_up_vis, s_map], dim=-1).permute(0, 2, 3, 1).squeeze(0)
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(vis).save("vis.png")



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
    ):
        super(DINOv3_Encoder, self).__init__()
        
        self.model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=pretrained_model_name_or_path)
        self.model.eval().to(device)
        self.device = device
        self.antialias = antialias
        self.dtype = torch.bfloat16

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)
        self.size = size
        if freeze:
            self.freeze()


    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
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
    


image_path = "datasets/kubric_DELTA/movif/kubric_processed_mix_3d/0000/frames/000.png"
image = Image.open(image_path)

if False:
    pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path, 
        device_map="auto", 
    )
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)
elif False:
    import torchvision
    from torchvision.transforms import v2
    def make_transform(resize_size: int = 256):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])
    trans = make_transform()
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=pretrained_model_name_or_path)
    image_th = trans(image)


    import pdb
    pdb.set_trace()
    # ['x_norm_clstoken', 'x_storage_tokens', 'x_norm_patchtokens', 'x_prenorm', 'masks']
    ret = dinov3_vitl16.forward_features(image_th.unsqueeze(0))
elif True:
    model = DINOv3_Encoder()
    model.to("cuda")
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")

    feats = model(image_th)

    vis_dino(feats, image_th)

    import pdb
    pdb.set_trace()
