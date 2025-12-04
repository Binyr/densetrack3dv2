from pathlib import Path
f_path = Path(__file__)
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import cv2
import numpy as np
import numpy as np
import time
import viser
import torch

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


def tensor_to_numpy(x):
    """支持 torch.Tensor / np.ndarray."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def backproject_uvd_to_cam(u, v, depth, K):
    """
    u, v, depth 可以是标量或同形状数组，K: (3,3)
    返回 cam 坐标 (.., 3)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth
    return np.stack([X, Y, Z], axis=-1)


def transform_cam_to_world(points_cam, ext_4x4):
    """
    points_cam: (N,3)
    ext_4x4: (4,4), 假设是 cam2world
    返回 (N,3) world 坐标
    """
    N = points_cam.shape[0]
    homo = np.concatenate([points_cam, np.ones((N, 1), dtype=points_cam.dtype)], axis=-1)  # (N,4)
    world_h = (ext_4x4 @ homo.T).T  # (N,4)
    # 一般最后一维是 1，这里还是保险除一下
    world = world_h[:, :3] / np.clip(world_h[:, 3:4], 1e-8, None)
    return world


def make_rgb_uint8(img):
    """
    img: (H,W,3)，任意 float / uint8
    输出 uint8 [0,255]
    """
    if img.dtype == np.uint8:
        return img
    # float
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


import os
from plyfile import PlyData, PlyElement
import json

def save_ply(points, colors, path):
    """
    points: (N,3)
    colors: (N,3) uint8
    """
    assert points.shape[0] == colors.shape[0]

    N = points.shape[0]
    verts = []

    for i in range(N):
        x, y, z = points[i]
        r, g, b = colors[i]
        verts.append((x, y, z, r, g, b))

    vertex_dtype = [('x', 'f4'), ('y','f4'), ('z','f4'),
                    ('red','u1'), ('green','u1'), ('blue','u1')]

    ply = PlyData([PlyElement.describe(np.array(verts, dtype=vertex_dtype), 'vertex')], text=True)
    ply.write(path)

def export_viser_data(
    out_dir,
    all_frame_points,
    all_frame_colors,
    track_points,
    track_colors,
    intrinsics,
    extrinsics,
):
    """
    all_frame_points: list of (Ni,3) arrays
    all_frame_colors: list of (Ni,3) arrays
    track_points: list of (Li,3)
    track_colors: list of (Li,3)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. save frame point clouds
    for t, (pts, cols) in enumerate(zip(all_frame_points, all_frame_colors)):
        save_ply(pts, cols, os.path.join(out_dir, f"frame_{t:03d}.ply"))

    # 2. save trajectory points
    if len(track_points) > 0:
        track_pts = np.concatenate(track_points, axis=0)
        track_cols = np.concatenate(track_colors, axis=0)
        save_ply(track_pts, track_cols, os.path.join(out_dir, "trajectories.ply"))

    # 3. save intrinsics & extrinsics
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(out_dir, "extrinsics.npy"), extrinsics)

    print(f"✔ Export finished. Files saved in: {out_dir}")

def export_scene_single_file(
    save_path,
    all_frame_points,
    all_frame_colors,
    all_track_points,
    all_track_colors,
    intrinsics,
    extrinsics,
):
    """
    Save the entire visualization into a single .npz file.
    """
    # Concatenate frames into a ragged list
    frame_points = [p.astype(np.float32) for p in all_frame_points]
    frame_colors = [c.astype(np.uint8)   for c in all_frame_colors]

    # Trajectories
    track_points = [p.astype(np.float32) for p in all_track_points]
    track_colors = [c.astype(np.uint8)   for c in all_track_colors]

    np.savez_compressed(
        save_path,
        frame_points=frame_points,
        frame_colors=frame_colors,
        track_points=track_points,
        track_colors=track_colors,
        intrinsics=intrinsics.astype(np.float32),
        extrinsics=extrinsics.astype(np.float32),
    )

    print(f"✔ Scene saved to single file: {save_path}")

def visualize_with_viser(
    rgb,
    depth,
    traj_uv,
    traj_d,
    intrinsics,
    extrinsics,
    stride: int = 4,
    save_path = "vis/viser.ply"
):
    """
    用 viser 可视化点云 + 轨迹点.
    stride: 为了减少点数，对深度图做下采样（每 stride 像素取一点）
    """
    rgb = tensor_to_numpy(rgb)           # (T,3,H,W)
    depth = tensor_to_numpy(depth)       # (T,1,H,W)
    traj_uv = tensor_to_numpy(traj_uv)   # (T,N,2)
    traj_d = tensor_to_numpy(traj_d)     # (T,N,1)
    intrinsics = tensor_to_numpy(intrinsics)   # (T,3,3)
    extrinsics = tensor_to_numpy(extrinsics)   # (T,4,4)

    T, C, H, W = rgb.shape
    _, N, _ = traj_uv.shape

    # 如果是 world2cam，可以在这里反一下：
    # extrinsics = np.linalg.inv(extrinsics)

    server = viser.ViserServer()
    scene = server.scene

    # 可选：加一个世界坐标系
    scene.add_frame(name="/world_frame")


    all_frame_points = []
    all_frame_colors = []
    all_track_points = []
    all_track_colors = []

    # -------------------
    # 1) 每帧点云
    # -------------------
    v_idx = np.arange(0, H, stride)
    u_idx = np.arange(0, W, stride)
    uu, vv = np.meshgrid(u_idx, v_idx)   # (H_s, W_s)

    for t in range(T):
        K = intrinsics[t]      # (3,3)
        ext = extrinsics[t]    # (4,4)

        # 深度采样
        d = depth[t, 0]        # (H,W)
        d_sample = d[vv, uu]   # (H_s, W_s)

        # 反投影到相机坐标
        pts_cam = backproject_uvd_to_cam(uu, vv, d_sample, K)  # (H_s, W_s, 3)
        pts_cam = pts_cam.reshape(-1, 3)

        # 有些深度可能是 0 或 nan，过滤一下
        valid = np.isfinite(pts_cam).all(axis=-1) & (pts_cam[:, 2] > 0)
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] == 0:
            continue

        pts_world = transform_cam_to_world(pts_cam, ext)  # (N,3)

        # 颜色
        img = rgb[t].transpose(1, 2, 0)  # (H,W,3)
        img_vis = make_rgb_uint8(img)
        cols = img_vis[vv, uu].reshape(-1, 3)[valid]

        # scene.add_point_cloud(
        #     name=f"/frames/frame_{t:03d}/point_cloud",
        #     points=pts_world,
        #     colors=cols.astype(np.uint8),
        #     point_size=0.01,
        # )
        all_frame_points.append(pts_world)
        all_frame_colors.append(cols.astype(np.uint8))

    # -------------------
    # 2) 轨迹 3D 点（所有帧累积）
    # 每条轨迹一个颜色
    # -------------------
    rng = np.random.RandomState(0)
    track_colors = rng.randint(0, 255, size=(N, 3), dtype=np.uint8)

    for k in range(N):  # 第 k 条轨迹
        world_points = []

        for t in range(T):
            u, v = traj_uv[t, k]        # (2,)
            z = traj_d[t, k, 0]         # 标量

            if not np.isfinite(z) or z <= 0:
                continue

            K = intrinsics[t]
            ext = extrinsics[t]

            # 注意：u, v 是像素坐标（列、行）
            pt_cam = backproject_uvd_to_cam(u, v, z, K)[None, ...]  # (1,3)
            pt_world = transform_cam_to_world(pt_cam, ext)[0]       # (3,)
            world_points.append(pt_world)

        if len(world_points) == 0:
            continue

        world_points = np.stack(world_points, axis=0)  # (L,3)
        color = track_colors[k]
        colors = np.repeat(color[None, :], world_points.shape[0], axis=0)

        # scene.add_point_cloud(
        #     name=f"/tracks/track_{k:03d}",
        #     points=world_points,
        #     colors=colors,
        #     point_size=0.04,
        # )
        all_track_points.append(world_points)
        all_track_colors.append(colors)
    
    if save_path is not None:
        export_scene_single_file(
            save_path,
            all_frame_points,
            all_frame_colors,
            all_track_points,
            all_track_colors,
            intrinsics,
            extrinsics,
        )

    print("Viser server started, open http://localhost:8080 查看可视化")
    print("按 Ctrl+C 结束")

    # try:
    #     while True:
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass


# =======================
# 用法示例（伪代码）
# =======================
if __name__ == "__main__":
    # 这里用你自己的数据替换
    # rgb        = ...  # (24,3,H,W)
    # depth      = ...  # (24,1,H,W)
    # traj_uv    = ...  # (24,256,2)
    # traj_d     = ...  # (24,256,1)
    # intrinsics = ...  # (24,3,3)
    # extrinsics = ...  # (24,4,4)
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
            if False:
                rgbs = sample[0].video.numpy().astype(np.uint8)
                trajs = sample[0].trajectory.numpy()
                depths = sample[0].videodepth.numpy()
                draw_trajs_to_video_no_cv2(rgbs, trajs, depths, out_path=f"vis/PO/{i}.mp4")
            else:
                rgbs = sample[0].video
                trajs = sample[0].trajectory
                trajs_d = sample[0].trajectory_d
                distance = sample[0].videodepth
                intrinsics = sample[0].intrs
                cam2worlds = sample[0].cam2worlds
                depths = distance_to_depth(distance, intrinsics)

            visualize_with_viser(
                rgbs,
                depths,
                trajs,
                trajs_d,
                intrinsics,
                cam2worlds,
                stride=4,
            )
        # except:
        #     print(f"{i} data is error")
        
            import pdb
            pdb.set_trace()

    



