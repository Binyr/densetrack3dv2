from pathlib import Path
f_path = Path(__file__)
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import os
import time
import numpy as np
import torch
import viser

from plyfile import PlyData, PlyElement
from densetrack3d.datasets.point_odyssey_official import PointOdysseyDataset
from densetrack3d.utils.utils import distance_to_depth


# -------------------------
# 工具函数
# -------------------------
def tensor_to_numpy(x):
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
    homo = np.concatenate(
        [points_cam, np.ones((N, 1), dtype=points_cam.dtype)],
        axis=-1
    )  # (N,4)
    world_h = (ext_4x4 @ homo.T).T  # (N,4)
    world = world_h[:, :3] / np.clip(world_h[:, 3:4], 1e-8, None)
    return world


def make_rgb_uint8(img):
    """
    img: (H,W,3)，任意 float / uint8
    输出 uint8 [0,255]
    """
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def export_scene_single_file(
    save_path,
    all_frame_points,
    all_frame_colors,
    track_points_per_frame,
    track_colors_per_frame,
    intrinsics,
    extrinsics,
):
    """
    保存所有帧点云 + 每帧轨迹点 到一个 npz 文件.
    """
    frame_points = [p.astype(np.float32) for p in all_frame_points]
    frame_colors = [c.astype(np.uint8) for c in all_frame_colors]

    track_points_per_frame = [p.astype(np.float32) for p in track_points_per_frame]
    track_colors_per_frame = [c.astype(np.uint8) for c in track_colors_per_frame]

    np.savez_compressed(
        save_path,
        frame_points=frame_points,
        frame_colors=frame_colors,
        track_points_per_frame=track_points_per_frame,
        track_colors_per_frame=track_colors_per_frame,
        intrinsics=intrinsics.astype(np.float32),
        extrinsics=extrinsics.astype(np.float32),
    )
    print(f"✔ Scene saved to single file: {save_path}")


# -------------------------
# 1) 从 rgb/depth/traj... 生成点云 + 每帧轨迹，并导出 npz
# -------------------------
def visualize_with_viser(
    rgb,
    depth,
    traj_uv,
    traj_d,
    intrinsics,
    extrinsics,
    stride: int = 4,
    save_path="vis/viser_scene.npz",
):
    """
    不在这里起 viser，用于生成点云 & 每帧轨迹 并导出到 npz.
    """
    rgb = tensor_to_numpy(rgb)           # (T,3,H,W)
    depth = tensor_to_numpy(depth)       # (T,1,H,W)
    traj_uv = tensor_to_numpy(traj_uv)   # (T,N,2)
    traj_d = tensor_to_numpy(traj_d)     # (T,N,1)
    intrinsics = tensor_to_numpy(intrinsics)   # (T,3,3)
    extrinsics = tensor_to_numpy(extrinsics)   # (T,4,4)

    T, C, H, W = rgb.shape
    _, N, _ = traj_uv.shape

    # -------------------
    # 1) 每帧点云
    # -------------------
    all_frame_points = []
    all_frame_colors = []

    v_idx = np.arange(0, H, stride)
    u_idx = np.arange(0, W, stride)
    uu, vv = np.meshgrid(u_idx, v_idx)   # (H_s, W_s)

    for t in range(T):
        K = intrinsics[t]
        ext = extrinsics[t]

        d = depth[t, 0]                  # (H,W)
        d_sample = d[vv, uu]             # (H_s, W_s)

        pts_cam = backproject_uvd_to_cam(uu, vv, d_sample, K)  # (H_s,W_s,3)
        pts_cam = pts_cam.reshape(-1, 3)

        valid = np.isfinite(pts_cam).all(axis=-1) & (pts_cam[:, 2] > 0)
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] == 0:
            all_frame_points.append(np.zeros((0, 3), dtype=np.float32))
            all_frame_colors.append(np.zeros((0, 3), dtype=np.uint8))
            continue

        pts_world = transform_cam_to_world(pts_cam, ext)  # (N,3)

        img = rgb[t].transpose(1, 2, 0)  # (H,W,3)
        img_vis = make_rgb_uint8(img)
        cols = img_vis[vv, uu].reshape(-1, 3)[valid]

        all_frame_points.append(pts_world.astype(np.float32))
        all_frame_colors.append(cols.astype(np.uint8))

    # -------------------
    # 2) 每帧轨迹点（只显示当前帧上的点，不显示过去的尾巴）
    # -------------------
    rng = np.random.RandomState(0)
    per_track_color = rng.randint(0, 255, size=(N, 3), dtype=np.uint8)

    track_points_per_frame = []
    track_colors_per_frame = []

    for t in range(T):
        K = intrinsics[t]
        ext = extrinsics[t]

        pts_this_frame = []
        cols_this_frame = []

        for k in range(N):
            u, v = traj_uv[t, k]
            z = traj_d[t, k, 0]

            if not np.isfinite(z) or z <= 0:
                continue

            pt_cam = backproject_uvd_to_cam(u, v, z, K)[None, ...]  # (1,3)
            pt_world = transform_cam_to_world(pt_cam, ext)[0]       # (3,)

            pts_this_frame.append(pt_world)
            cols_this_frame.append(per_track_color[k])

        if len(pts_this_frame) == 0:
            track_points_per_frame.append(np.zeros((0, 3), dtype=np.float32))
            track_colors_per_frame.append(np.zeros((0, 3), dtype=np.uint8))
        else:
            pts_this_frame = np.stack(pts_this_frame, axis=0)
            cols_this_frame = np.stack(cols_this_frame, axis=0)
            track_points_per_frame.append(pts_this_frame.astype(np.float32))
            track_colors_per_frame.append(cols_this_frame.astype(np.uint8))

    # -------------------
    # 3) 保存到单个 npz 文件
    # -------------------
    os.makedirs(Path(save_path).parent, exist_ok=True)
    export_scene_single_file(
        save_path,
        all_frame_points,
        all_frame_colors,
        track_points_per_frame,
        track_colors_per_frame,
        intrinsics,
        extrinsics,
    )


# -------------------------
# 2) 加载 npz，用 slider 可视化 帧点云 + 帧轨迹
# -------------------------
def load_scene(npz_path, port=9090):
    data = np.load(npz_path, allow_pickle=True)

    frame_points = data["frame_points"]              # list, len T
    frame_colors = data["frame_colors"]
    track_points_per_frame = data["track_points_per_frame"]
    track_colors_per_frame = data["track_colors_per_frame"]

    T = len(frame_points)
    print(f"Loaded {T} frames from {npz_path}")

    server = viser.ViserServer(port=port)
    scene = server.scene

    scene.add_frame(name="/world")

    # GUI slider
    gui_frame = server.gui.add_slider(
        "Frame",     # label
        0,           # min
        T - 1,       # max
        1,           # step
        0            # initial value
    )

    # 动态帧点云
    pc_node = scene.add_point_cloud(
        name="/dynamic/frame_cloud",
        points=frame_points[0],
        colors=frame_colors[0],
        point_size=0.01,
    )

    # 动态轨迹点（当前帧）
    track_node = scene.add_point_cloud(
        name="/dynamic/frame_tracks",
        points=track_points_per_frame[0],
        colors=track_colors_per_frame[0],
        point_size=0.04,
    )

    print(f"✔ Scene loaded with slider. Open: http://localhost:{port}")

    last_t = -1
    while True:
        t = int(gui_frame.value)
        if t != last_t:
            pc_node.points = frame_points[t]
            pc_node.colors = frame_colors[t]

            track_node.points = track_points_per_frame[t]
            track_node.colors = track_colors_per_frame[t]

            last_t = t
        time.sleep(0.03)


# -------------------------
# 示例入口
# -------------------------
if __name__ == "__main__":
    """
    两种用法（二选一）:

    1) 从 PointOdyssey 里导出一个场景:
       python this_file.py export

    2) 加载已经导出的 npz 用 viser 看:
       python this_file.py vis path/to/viser_scene.npz
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python this_file.py export          # 从数据集中导出一个样本")
        print("  python this_file.py vis scene.npz   # 可视化该 npz")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "export":
        dataset = PointOdysseyDataset(
            dataset_location="datasets/point0dyssey/",
            dset="train",
            use_augs=False,
            S=24,
            N=768,
            crop_size=(384, 512),
            resize_size=(384 + 64, 512 + 64),
        )

        # 这里只导出第 0 个样本，你可以按需改成循环
        idx = 0
        sample = dataset[idx]

        rgbs = sample[0].video          # (T,3,H,W)
        trajs = sample[0].trajectory    # (T,N,2)
        trajs_d = sample[0].trajectory_d
        distance = sample[0].videodepth
        intrinsics = sample[0].intrs
        cam2worlds = sample[0].cam2worlds

        depths = distance_to_depth(distance, intrinsics)

        save_path = "vis/viser_scene.npz"
        visualize_with_viser(
            rgbs,
            depths,
            trajs,
            trajs_d,
            intrinsics,
            cam2worlds,
            stride=4,
            save_path=save_path,
        )
        print("Export done.")

    elif mode == "vis":
        if len(sys.argv) < 3:
            print("Usage: python this_file.py vis path/to/scene.npz")
            sys.exit(1)
        npz_path = sys.argv[2]
        load_scene(npz_path, port=9090)
    else:
        print("Unknown mode:", mode)
        sys.exit(1)
