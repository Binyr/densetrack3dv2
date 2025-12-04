from pathlib import Path
f_path = Path(__file__)
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import cv2
import numpy as np

from densetrack3d.datasets.point_odyssey_official import PointOdysseyDataset
from densetrack3d.utils.utils import distance_to_depth

import imageio
from PIL import Image, ImageDraw

# import utils.geom
# print(utils.geom.apply_4x4_py)

_DEFAULT_EXCLUDE_SCENES = {
    "character", "character0", "character0_", "character0_f", "character0_f2",
    "character1", "character1_f",
    "character2", "character2_", "character2_f",
    "character3", "character3_f",
    "character4", "character4_", "character4_f",
    "character5", "character5_", "character5_f",
    "character6", "character6_f",
    "gso_in_big", "gso_out_big",
}

def draw_trajs_to_video_no_cv2(rgbs, trajs, depths, out_path="traj_vis.mp4", fps=10):
    """
    rgbs : (T, 3, H, W)
    trajs: (T, N, 2)
    """
    T, _, H, W = rgbs.shape
    N = trajs.shape[1]

    # convert to uint8 (H,W,3)
    if rgbs.dtype != np.uint8:
        rgbs = (rgbs * 255).clip(0, 255).astype(np.uint8)
    rgbs = np.transpose(rgbs, (0, 2, 3, 1))  # -> (T, H, W, 3)

    # === normalize depth to [0,255] for color ===
    depth_map = depths[:, 0]  # (T,H,W)
    d_min = np.percentile(depth_map, 1)
    d_max = np.percentile(depth_map, 99)
    depth_norm = ((depth_map - d_min) / (d_max - d_min + 1e-6))
    depth_norm = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    depth_norm = depth_norm[..., None].repeat(3, 3) # -> (T, H, W, 3)

    rgbs = np.concatenate([rgbs, depth_norm], axis=2)
    # random colors for N tracks
    colors = np.random.randint(0, 255, (N, 3)).tolist()

    writer = imageio.get_writer(out_path, fps=fps)

    for t in range(T):
        # use PIL to draw
        img = Image.fromarray(rgbs[t])
        draw = ImageDraw.Draw(img)

        for i in range(N):
            x, y = trajs[t, i]

            # draw point
            if 0 <= x < W and 0 <= y < H:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=tuple(colors[i]))

            # draw line from previous frame
            if t > 0:
                x_prev, y_prev = trajs[t-1, i]
                if (0 <= x_prev < W and 0 <= y_prev < H and
                    0 <= x < W and 0 <= y < H):
                    draw.line([(x_prev, y_prev), (x, y)], fill=tuple(colors[i]), width=1)

        # write frame
        writer.append_data(np.array(img))

    writer.close()
    print("Saved to", out_path)

dataset = PointOdysseyDataset(
    dataset_location="datasets/point0dyssey/",
    dset="train",
    use_augs=False,
    S=24,
    N=768,
    crop_size=(384, 512),
    resize_size=(384+64, 512+64),
)

for i in range(len(dataset)):
    # try:
        sample = dataset.__getitem__(i)
        rgbs = sample[0].video.numpy().astype(np.uint8)
        trajs = sample[0].trajectory.numpy()
        depths = sample[0].videodepth.numpy()
        draw_trajs_to_video_no_cv2(rgbs, trajs, depths, out_path=f"vis/PO/{i}.mp4")
    # except:
    #     print(f"{i} data is error")

    
import pdb
pdb.set_trace()