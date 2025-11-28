import torch
import torch.nn as nn
from einops import rearrange
from submodules.Pi3.pi3.models.pi3 import Pi3

REPOV3_DIR = "/mnt/shared-storage-user/binyanrui/Projects/dinov3"
DINOV3_DIR = "/mnt/shared-storage-user/si-data/DINOv3/"
dinov3_vitl16_weight_path = f"{DINOV3_DIR}dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
dinov3_vitb16_weight_path = f"{DINOV3_DIR}dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

REPOV2_DIR = "/mnt/shared-storage-user/binyanrui/Projects/dinov2"
DINOV2_DIR = "/mnt/shared-storage-user/idc2-shared/binyanrui/pretrained_models/dino/v2/"
dinov2_vitl14_weight_path = f"{DINOV2_DIR}dinov2_vitl14_pretrain.pth"
dinov2_vitb14_weight_path = f"{DINOV2_DIR}dinov2_vitb14_pretrain.pth"

NAME_TO_PATCH = {
    # v3
    "dinov3_vitl16": (REPOV3_DIR, DINOV3_DIR, dinov3_vitl16_weight_path, 16, 1024),
    "dinov3_vitb16": (REPOV3_DIR, DINOV3_DIR, dinov3_vitb16_weight_path, 16, 768),
    # v2
    "dinov2_vitl14": (REPOV2_DIR, DINOV2_DIR, dinov2_vitl14_weight_path, 14, 1024),
    "dinov2_vitb14": (REPOV2_DIR, DINOV2_DIR, dinov2_vitb14_weight_path, 14, 768),

}

class Pi3_Encoder(torch.nn.Module):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name = 'dinov3_vitl16',
        freeze = False,
        antialias=True,
        device="cuda",
        patch_size = (64, 64), # H, W
        use_deconv1 = False,
        use_deconv2 = False,
        dtype=torch.bfloat16
    ):
        super(Pi3_Encoder, self).__init__()
        
        self.model = Pi3.from_pretrained("/mnt/shared-storage-user/idc2-shared/binyanrui/pretrained_models/Pi3").to(device).eval()
        self.model.eval().to(device).to(dtype)
        self.stride = 14
        self.C = 2048

        stride = self.stride
        C = self.C

        self.device = device
        self.antialias = antialias
        self.dtype = dtype

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)

        
        self.patch_size = patch_size
        self.size = [x * stride for x in patch_size]
        if freeze:
            self.freeze()

        # transpose conv
        self.use_deconv1 = use_deconv1
        self.use_deconv2 = use_deconv2
        if use_deconv1:
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(4)
            in_channels1 = C
            out_channels1 =  C // 4
            self.deconv1 = nn.ConvTranspose2d(
                in_channels=in_channels1,
                out_channels=out_channels1,
                kernel_size=deconv_kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False
            )
            torch.nn.init.xavier_uniform_(self.deconv1.weight, gain=1.0) # nonlinearity="conv_transpose2d")
            self.C = out_channels1

        if use_deconv2:
            assert use_deconv1 == True
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(4)
            in_channels2 = out_channels1
            out_channels2 =  out_channels1 // 2
            self.deconv2 = nn.ConvTranspose2d(
                in_channels=in_channels2,
                out_channels=out_channels2,
                kernel_size=deconv_kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False
            )
            
            torch.nn.init.xavier_uniform_(self.deconv2.weight, gain=1.0) # nonlinearity="conv_transpose2d")
            self.C = out_channels2

    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward_deconv(self, x):
        if self.use_deconv1:
            x = self.deconv1(x)
        if self.use_deconv2:  
            x = self.deconv2(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def encoder(self, x, size=None, return_every=False):
        '''
        x: [b t h w c], range from (-1, 1), rbg
        '''
        ori_dtype = x.dtype
        self.to(self.dtype)
        x = x.to(self.dtype)
        x = self.preprocess(x, size).to(self.device, self.dtype)

        b, t, c, h, w = x.shape
        patch_h, patch_w = h // self.stride, w // self.stride

        embeddings_lis = self.model.encode_image(x, return_every=return_every)

        if not isinstance(embeddings_lis, list):
            embeddings_lis = [embeddings_lis]
        
        for i in range(len(embeddings_lis)):
            embeddings = embeddings_lis[i]

            embeddings = rearrange(embeddings, 'b t h w c -> (b t) c h w')

            if self.use_deconv1 or self.use_deconv2:
                embeddings = self.forward_deconv(embeddings)

            embeddings = embeddings.to(ori_dtype)
            embeddings = rearrange(embeddings, '(b t) c h w -> b t c h w', b=b)

            embeddings_lis[i] = embeddings
        return embeddings_lis

    def preprocess(self, x, size=None):
        ''' x
        '''
        # normalize to [0,1],
        if size is None:
            if isinstance(self.size, int):
                size = (self.size, self.size)
            else:
                size = self.size
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = torch.nn.functional.interpolate(
            x,
            size=size,
            mode='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        # to [0, 1]
        x = (x + 1.0) / 2.0

        return x
    
    def to(self, device, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.model.to(device, dtype)
            self.mean.to(device, dtype)
            self.std.to(device, dtype)
            if self.use_deconv1:
                self.deconv1.to(device, dtype)
            if self.use_deconv2:
                self.deconv2.to(device, dtype)
        else:
            self.model.to(device)
            self.mean.to(device)
            self.std.to(device)
            if self.use_deconv1:
                self.deconv1.to(device)
            if self.use_deconv2:
                self.deconv2.to(device)
        return self

    def forward(self, x, **kwargs):
        return self.encoder(x, **kwargs)