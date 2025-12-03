from pathlib import Path
f_path = Path(__file__).absolute()
import sys
sys.path.insert(0, str(f_path.parents[1]))

import torch

from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

from densetrack3d.models.densetrack3d.blocks import BasicEncoder, Mlp
from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2

def vis_dino(feats_l, image, save_path):
    _, _, H, W = image.shape
    vis_l = []
    for i, feats in enumerate(feats_l):
        _, _, FH, FW = feats.shape
        # selected_point = [H // 2, W // 2]
        selected_point = [90, 400]
        feats_up = torch.nn.functional.interpolate(feats, size=(H, W), mode="bilinear")
        feats_up_vis = feats_up[:, :3]
        feats_up_vis = (feats_up_vis - feats_up_vis.min()) / (feats_up_vis.max() - feats_up_vis.min())
        image_ = image * 0.5 + 0.5

        # similarity map
        # s_map = (feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1) * feats_up)
        s_map = torch.nn.functional.cosine_similarity(feats_up[:, :, selected_point[0], selected_point[1]].unsqueeze(-1).unsqueeze(-1), feats_up, dim=1)
        print(s_map.max(), s_map.min(), FH, FW)
        s_map = s_map.unsqueeze(1).repeat(1,3,1,1)
        s_map = (s_map + 1.) / 2.
        
        vis = torch.cat([image_, feats_up_vis, s_map], dim=-1).permute(0, 2, 3, 1).squeeze(0)
        vis = (vis.cpu().numpy() * 255).astype(np.uint8)
        if i == 0:
            r = 3*4
        elif i == 1:
            r = 3*2
        elif i == 2:
            r = 7 * 8
        else:
            raise
        
        top_left = (2*W + selected_point[1] - r, selected_point[0] - r)
        bottom_right = (2*W + selected_point[1] + r, selected_point[0] + r)
        vis = np.ascontiguousarray(vis)
        vis = cv2.rectangle(vis, top_left, bottom_right, (255, 0, 0), 2)
        vis_l.append(vis)
    vis_l = np.concatenate(vis_l, axis=0)
    Image.fromarray(vis_l).save(save_path)

model = DenseTrack3DV2(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        coarse_to_fine_dense=True,
        use_dino="dinov3_vitl16",
        dino_size=32,
        # 
        use_deconv=True,
        use_merge_conv=True,
        #
        radius_supp="[3,3,7]",
        stride_supp="[1,1,1]",
        radius_corr="[3,3,7]",
        stride_corr="[1,1,1]",
    )
state_dict = torch.load("logdirs/densetrack3dv2_ft_lr5e-5_4w_corr_v2_dinov3_vitl16/model_densetrack3dv2_final.pth")
new_state_dict = state_dict["model"]

miss, unexpected = model.load_state_dict(new_state_dict)
print(f"miss: {miss}")
print(f"unexpected: {unexpected}")

dataset_root = Path("datasets/kubric_DELTA/movif/kubric_processed_mix_3d_instance/")
for seq_id in tqdm(range(100)):
    seq_name = f"{seq_id:04d}"
    seq_path = dataset_root / seq_name
    for img_id in range(0, 24, 5):
        img_name = f"{img_id:03d}.png"
        img_path = seq_path / "frames" / img_name

        img_path = "datasets/kubric_DELTA/movif/kubric_processed_mix_3d/0000/frames/000.png"
        image = Image.open(img_path)

        model.to("cuda")
        image_th = torch.from_numpy(np.array(image))
        image_th = image_th / 255. * 2. - 1.
        image_th = image_th.permute(2, 0, 1).unsqueeze(0).to("cuda").unsqueeze(1)
        with torch.no_grad():
            feats, feats_high_res, feats_low_res, _ = model.extract_features(image_th)
            feats, feats_high_res, feats_low_res = feats[:, 0], feats_high_res[:, 0], feats_low_res[:, 0]
        
        save_path = Path("vis") / "kubric" / "deltav2" / seq_name / img_name
        save_path = Path("vis") / "kubric2" / "deltav2_corrv2_dinov3_vitl16" / seq_name / img_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis_dino([feats, feats_high_res, feats_low_res], image_th[:, 0], str(save_path))
        # raise

