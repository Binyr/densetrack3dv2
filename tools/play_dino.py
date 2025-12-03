from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]

import sys
sys.path.insert(0, str(p_path))

from densetrack3d.utils.dino_encoder import DINO_Encoder

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from einops import rearrange
from densetrack3d.utils.chrono_track_model import LocoTrack
from densetrack3d.utils.Pi3_encoder import Pi3_Encoder

from PIL import Image
import numpy as np
import pdb
import cv2

def vis_dino(feats, image, name="vis_dino2.png"):
    _, _, H, W = image.shape
    _, _, FH, FW = feats.shape
    selected_point = [H // 2, W // 2]
    selected_point = [90, 400]
    selected_point = [37, 202]
    selected_point = [37, 192]
    # selected_point = [37, 182]
    print(feats.shape)
    feats_up = torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear")
    feats_up_vis = feats_up[:, :3]
    feats_up_vis = (feats_up_vis - feats_up_vis.min()) / (feats_up_vis.max() - feats_up_vis.min())
    image = image * 0.5 + 0.5

    # similarity map
    # s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up)
    # s_map = torch.nn.functional.cosine_similarity(feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1), feats_up, dim=1)
    s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up).sum(dim=1)
    s_map = (s_map / s_map.abs().max())
    # import pdb
    # pdb.set_trace()
    print(s_map.max(), s_map.min())
    s_map = s_map.unsqueeze(1).repeat(1,3,1,1)
    s_map = (s_map + 1.) / 2.
    
    vis = torch.cat([image, feats_up_vis, s_map], dim=-1).permute(0, 2, 3, 1).squeeze(0)
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)
    vis = cv2.UMat(vis)
    vis = cv2.circle(vis, center=(2*W + selected_point[1], selected_point[0]), radius=14, color=(255, 0, 0))
    vis = cv2.circle(vis, center=(selected_point[1], selected_point[0]), radius=14, color=(255, 0, 0))
    
    Image.fromarray(vis.get()).save(name)


def vis_dino2(feats, image, name="vis_dino2.png"):
    _, _, H, W = image.shape
    _, _, FH, FW = feats.shape
    selected_point = [H // 2, W // 2]
    selected_point = [90, 400]
    selected_point = [37, 202]
    selected_point = [37, 192]
    # selected_point = [37, 182]
    print(feats.shape)
    feats_up = torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear")
    feats_up_vis = feats_up[:, :3]
    feats_up_vis = (feats_up_vis - feats_up_vis.min()) / (feats_up_vis.max() - feats_up_vis.min())
    image = image * 0.5 + 0.5

    # similarity map
    # s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up)
    # s_map = torch.nn.functional.cosine_similarity(feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1), feats_up, dim=1)
    s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up).sum(dim=1)
    s_map = (s_map / s_map.abs().max())
    # import pdb
    # pdb.set_trace()
    print(s_map.max(), s_map.min())
    s_map = s_map.unsqueeze(1).repeat(1,3,1,1)
    s_map = (s_map + 1.) / 2.
    
    vis = torch.cat([image, s_map], dim=-2).permute(0, 2, 3, 1).squeeze(0)
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)
    # vis = cv2.UMat(vis)
    # vis = cv2.circle(vis, center=(2*W + selected_point[1], selected_point[0]), radius=14, color=(255, 0, 0))
    # vis = cv2.circle(vis, center=(selected_point[1], selected_point[0]), radius=14, color=(255, 0, 0))
    r = 3*4
    top_left = (selected_point[1] - r, H + selected_point[0] - r)
    bottom_right = (selected_point[1] + r, H + selected_point[0] + r)
    vis = np.ascontiguousarray(vis)
    vis = cv2.rectangle(vis, top_left, bottom_right, (255, 0, 0), 2)
    top_left = (selected_point[1] - r, selected_point[0] - r)
    bottom_right = (selected_point[1] + r, selected_point[0] + r)
    vis = cv2.rectangle(vis, top_left, bottom_right, (255, 0, 0), 2)
    
    Image.fromarray(vis).save(name)


def vis_dino_multi_frame(feats_lis, image_lis, name="vis_dino2.png"):
    _, _, H, W = image_lis[0].shape
    _, _, FH, FW = feats_lis[0].shape
    selected_point = [H // 2, W // 2]
    selected_point = [90, 400]
    selected_point = [37, 202]
    selected_point = [37, 192]
    # selected_point = [37, 182]
    
    feats_lis_up = [torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear") for feats in feats_lis]

    feats_concat = torch.cat(feats_lis_up, dim=-1)
    # similarity map
    # s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up)
    # s_map = torch.nn.functional.cosine_similarity(feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1), feats_up, dim=1)
    s_map = (feats_concat[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_concat).sum(dim=1)
    s_map = (s_map / s_map.abs().max())
    print(s_map.max(), s_map.min())
    s_map = s_map.unsqueeze(1).repeat(1,3,1,1)
    s_map = (s_map + 1.) / 2.

    images_concat = torch.cat(image_lis, dim=-1)
    images_concat = images_concat * 0.5 + 0.5
    
    vis = torch.cat([images_concat, s_map], dim=-2).permute(0, 2, 3, 1).squeeze(0)
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)
    
    vis = np.ascontiguousarray(vis)
    r = 3*4
    top_left = (selected_point[1] - r, H + selected_point[0] - r)
    bottom_right = (selected_point[1] + r, H + selected_point[0] + r)
    vis = cv2.rectangle(vis, top_left, bottom_right, (255, 0, 0), 2)

    top_left = (selected_point[1] - r, selected_point[0] - r)
    bottom_right = (selected_point[1] + r, selected_point[0] + r)
    vis = cv2.rectangle(vis, top_left, bottom_right, (255, 0, 0), 2)
    
    Image.fromarray(vis).save(name)


image_path = "datasets/kubric_DELTA/movif/kubric_processed_mix_3d/0000/frames/000.png"
image = Image.open(image_path)

model_name = "dino_up3_multi_view"
# model_name = "pi3"

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
elif model_name == "dino_up2":
    model = DINO_Encoder(
        model_name="dinov3_vitl16",
        patch_size=(64, 64)
        # model_name="chrono"
    )
    model.to("cuda")
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feats = model(image_th, size=(4096, 4096))
        print(feats.shape)
    vis_dino(feats, image_th, name="vis/dinov3_vitl16.png")
elif model_name == "dino_up":
    default_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).to("cuda")
    default_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).to("cuda")
    model = DINO_Encoder(
        model_name="dinov3_vitl16",
        patch_size=(64, 64)
        # model_name="chrono"
    )
    upsampler = torch.hub.load(repo_or_dir="/mnt/shared-storage-user/binyanrui/Projects/anyup", model='anyup', source="local")
    upsampler = upsampler.to("cuda")

    model.to("cuda")
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feats = model(image_th, size=(512, 512))
        # for anyup
        image_th_anyup = image_th * 0.5 + 0.5
        image_th_anyup = (image_th_anyup - default_mean) / default_std
        feats = upsampler(image_th_anyup, feats)
        print(feats.shape)
    vis_dino(feats, image_th, name="vis/dinov3_vitl16_32_up.png")

elif model_name == "dino_up3":
    default_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).to("cuda")
    default_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).to("cuda")
    model = DINO_Encoder(
        model_name="dinov3_vitl16",
        patch_size=(64, 64),
        use_anyup=True,
        # model_name="chrono"
    )
    model.to("cuda")
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        feats = model(image_th, size=(512, 512))
        # for anyup
        print(feats.shape)
    vis_dino2(feats, image_th, name="vis/dinov3_vitl16_32_up3.png")

elif model_name == "chrono":
    model_kwargs = {"dino_size": "base", "dino_reg": False, "adapter_intermed_channels": 128}
    dino_net = LocoTrack(**model_kwargs)
    state_dict = torch.load("/mnt/shared-storage-user/idc2-shared/binyanrui/pretrained_models/chrono_tracking/chrono_base.ckpt", map_location=torch.device('cpu'))
    state_dict = state_dict["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in ["model.occ_linear.weight", "model.occ_linear.bias"]:
            continue
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    missing_keys, unexpected_keys = dino_net.load_state_dict(state_dict, strict=False)
    print(f"unexpected_keys: {unexpected_keys}")
    print(f"missing_keys: {missing_keys}")

    model = dino_net
    model.to("cuda")
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
    image_th = image_th.repeat(16, 1, 1, 1).unsqueeze(0) # 1, 16, c, h ,w
    with torch.no_grad():
        feats = model.forward_dino(image_th, img_mult=32)
    feats = feats[:, 0]
    image_th = image_th[:, 0]
    # import pdb
    # pdb.set_trace()
    vis_dino(feats, image_th)
elif model_name == "pi3":
    model = Pi3_Encoder(patch_size=(128,128))
    image_th = torch.from_numpy(np.array(image))
    image_th = image_th / 255. * 2. - 1.
    image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
    image_th = image_th.repeat(16, 1, 1, 1).unsqueeze(0) # 1, 16, c, h ,w
    with torch.no_grad():
        feats_lis = model(image_th, return_every=True)
    
    image_th = image_th[:, 0].clone()
    for i, feats in enumerate(feats_lis):
        feats = feats[:, 0]
        vis_dino(feats, image_th, name=f"vis/pi3_{i}_layer.png")

elif model_name == "perception_model":
    import submodules.perception_models.core.vision_encoder.pe as pe
    import submodules.perception_models.core.vision_encoder.transforms as transforms

elif model_name == "dino_up3_multi_view":
    # define model
    default_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).to("cuda")
    default_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).to("cuda")
    model = DINO_Encoder(
        model_name="dinov3_vitl16",
        patch_size=(32, 32),
        use_anyup=True,
        # model_name="chrono"
    )
    model.to("cuda")

    # define image
    image_dir = "datasets/kubric_DELTA/movif/kubric_processed_mix_3d/0000/frames/"
    image_paths = [f"{image_dir}{x:03d}.png" for x in range(0, 24, 6)]
    image_th_lis, feats_lis = [], []
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image_th = torch.from_numpy(np.array(image))
        image_th = image_th / 255. * 2. - 1.
        image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            feats = model(image_th)
            # for anyup
            print(i, feats.shape)

        image_th_lis.append(image_th)
        feats_lis.append(feats)
    
    vis_dino_multi_frame(feats_lis, image_th_lis, name="vis/dinov3_vitl16_32_anyup_multi_frame.png")