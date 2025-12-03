# datasets/dynamic_replica/dataset.py
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import cv2

from datasets.base.base_dataset import BaseDataset


# --------- 深度读取 ---------

def _load_16bit_png_depth_as_f32(path: str) -> np.ndarray:
    """读取打包为 float16 的 16-bit PNG，返回 float32(H,W)。"""
    with Image.open(path) as depth_pil:
        arr_u16 = np.array(depth_pil, dtype=np.uint16)
        depth = np.frombuffer(arr_u16.tobytes(), dtype=np.float16).astype(np.float32)
        depth = depth.reshape(depth_pil.size[1], depth_pil.size[0])
    return depth


def _load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    """支持 .png(half packed) / .exr；做尺度与清理。"""
    if path.lower().endswith(".exr"):
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if d is None:
            raise FileNotFoundError(f"Failed to read EXR depth: {path}")
        d = d[..., 0].astype(np.float32)
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = _load_16bit_png_depth_as_f32(path)
    else:
        raise ValueError(f"Unsupported depth file: {path}")
    d = d * float(scale_adjustment)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d.astype(np.float32)


# --------- 相机变换（按官方注释保持一致） ---------

def _ndc_isotropic_to_pixel_K(viewpoint: Dict[str, Any], w: int, h: int) -> np.ndarray:
    """从 PyTorch3D 的 ndc_isotropic 转到像素系 K。"""
    s = min(h, w)
    fx_ndc, fy_ndc = viewpoint["focal_length"]
    px_ndc, py_ndc = viewpoint["principal_point"]
    fx = fx_ndc * s / 2.0
    fy = fy_ndc * s / 2.0
    px = (-px_ndc * s / 2.0) + w / 2.0
    py = (-py_ndc * s / 2.0) + h / 2.0
    # 与上游对齐半像素
    px -= 0.5
    py -= 0.5
    K = np.array([[fx, 0.0, px],
                  [0.0, fy, py],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _pytorch3d_RT_to_w2c(viewpoint: Dict[str, Any]) -> np.ndarray:
    """从 PyTorch3D 的 R/T 得到 w2c，并做轴翻转；返回 4x4。"""
    R = np.array(viewpoint["R"], dtype=np.float32).T
    T = np.array(viewpoint["T"], dtype=np.float32)
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = flip @ R
    w2c[:3, 3]  = (flip @ T.reshape(3,)).astype(np.float32)
    return w2c


# --------- 数据集 ---------

class DynamicReplicaDataset(BaseDataset):
    """
    目录结构:
      data_root/
        {mode}/
          frame_annotations_{mode}.jgz
          <sequence_name>/...
    """

    def __init__(self,
                 data_root: str,
                 only_first_n_seqs: Optional[int] = None,
                 dynamic_filter_threshold: float = 0.0,
                 keep_static_prob: float = 1.0,
                 min_visible_frames: int = 2,
                 target_num: int = 6000,
                 enforce_sequential: bool = True,
                 **kwargs):
        if enforce_sequential:
            kwargs['shuffle'] = False
        super().__init__(**kwargs)

        self.dataset_label = "DynamicReplica"
        self.data_dir = osp.join(data_root, self.mode)

        self.dynamic_filter_threshold = float(dynamic_filter_threshold)
        self.keep_static_prob = float(keep_static_prob)
        self.min_visible_frames = int(min_visible_frames)
        self.target_num = int(target_num)

        # 索引序列与帧
        ann_path = osp.join(self.data_dir, f"frame_annotations_{self.mode}.jgz")
        import gzip, json
        with gzip.open(ann_path, "rt", encoding="utf-8") as f:
            ann_list = json.load(f)

        seq_to_frames: Dict[str, List[Dict[str, Any]]] = {}
        skipped = set()
        for a in ann_list:
            if a.get("camera_name", "left") != "left":
                continue
            img_rel = a["image"]["path"]
            if not osp.exists(osp.join(self.data_dir, img_rel)):
                skipped.add(a["sequence_name"])
                continue
            seq_to_frames.setdefault(a["sequence_name"], []).append(a)

        seq_names = sorted(seq_to_frames.keys())
        if only_first_n_seqs is not None:
            seq_names = seq_names[:only_first_n_seqs]
        self.sequences: List[str] = seq_names
        self.seq_frames: Dict[str, List[Dict[str, Any]]] = {}
        self.num_imgs: Dict[str, int] = {}

        for s in self.sequences:
            frames = sorted(seq_to_frames[s], key=lambda x: int(x["frame_number"]))
            self.seq_frames[s] = frames
            self.num_imgs[s] = len(frames)

        print(f"[{self.dataset_label}] Found {len(self.sequences)} sequences in {self.data_dir}", flush=True)
        if len(skipped) > 0:
            print(f"[{self.dataset_label}] Skipped sequences due to missing images: {sorted(list(skipped))}", flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    @staticmethod
    def _dynamic_filter(traj3d_TN3: np.ndarray,
                        threshold: float,
                        keep_static_prob: float,
                        rng: np.random.Generator):
        if threshold <= 0.0 and keep_static_prob >= 1.0:
            return None
        if traj3d_TN3.size == 0 or traj3d_TN3.shape[1] == 0:
            return None
        max_xyz = np.max(np.abs(traj3d_TN3), axis=0)
        min_xyz = np.min(np.abs(traj3d_TN3), axis=0)
        diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
        mask = diff_l2 > float(threshold)
        if keep_static_prob < 1.0:
            keep_mask = rng.random(mask.shape[0], dtype=np.float32) < float(keep_static_prob)
            mask = mask | keep_mask
        return mask

    def _get_views(self, index: int, resolution: Tuple[int, int], rng: np.random.Generator):
        scene = self.sequences[index]
        frames = self.seq_frames[scene]
        num_imgs = self.num_imgs[scene]

        F = int(self.frame_num) if self.frame_num > 0 else 1
        if num_imgs >= F:
            start = int(rng.integers(0, num_imgs - F + 1))
            idxs = list(range(start, start + F))
        else:
            idxs = list(range(num_imgs))
            if num_imgs > 0:
                idxs += [num_imgs - 1] * (F - num_imgs)
            else:
                idxs = [0] * F

        self.this_views_info = dict(scene=scene, pairs=idxs)

        # 串行读取与处理
        rgb_list: List[Image.Image] = []
        depth_list: List[np.ndarray] = []
        K_list: List[np.ndarray] = []
        c2w_list: List[np.ndarray] = []
        size_list: List[Tuple[int, int]] = []

        t2d_list: List[np.ndarray] = []
        t3d_list: List[np.ndarray] = []
        vis_raw_list: List[np.ndarray] = []

        # 来自 BaseDataset 的增广标志用于 fast path
        aug_crop = bool(getattr(self, "aug_crop", False))
        aug_focal = bool(getattr(self, "aug_focal", False))

        for local_t, global_t in enumerate(idxs):
            fa = frames[global_t]

            # 轨迹与深度
            traj_path = osp.join(self.data_dir, fa["trajectories"]["path"])
            traj = torch.load(traj_path, map_location="cpu", weights_only=False)

            depth_path = osp.join(self.data_dir, fa["depth"]["path"])
            scale_adj = fa["depth"]["scale_adjustment"]
            depth_raw = _load_depth(depth_path, scale_adj)

            # 图像
            img_np = traj["img"].numpy().astype(np.uint8)  # H,W,3
            h0, w0 = img_np.shape[0], img_np.shape[1]

            # 相机
            K_src = _ndc_isotropic_to_pixel_K(fa["viewpoint"], w0, h0)
            w2c   = _pytorch3d_RT_to_w2c(fa["viewpoint"])
            c2w   = np.linalg.inv(w2c).astype(np.float32)

            # 轨迹 (像素坐标) / 可见性
            t2d_src = traj["traj_2d"].numpy()[..., :2].astype(np.float32)      # (M0,2)
            t3d     = traj["traj_3d_world"].numpy().astype(np.float32)         # (M0,3)
            vis_raw = traj["verts_inds_vis"].numpy().astype(np.bool_)          # (M0,)

            if t3d.size == 0 and t2d_src.size > 0:
                t3d = np.zeros((t2d_src.shape[0], 3), dtype=np.float32)

            # fast path：尺寸匹配且无增广，则不进几何流程
            if (w0, h0) == tuple(resolution) and (not aug_crop) and (not aug_focal):
                rgb_pil = Image.fromarray(img_np, mode="RGB")
                depth = depth_raw.astype(np.float32)
                K_out = K_src
                t2d = t2d_src
                W, H = rgb_pil.size
            else:
                rgb_pil = Image.fromarray(img_np, mode="RGB")
                rgb_pil, depth, K_out, t2d = self._crop_resize_if_necessary(
                    rgb_pil, depth_raw, K_src, resolution, rng=rng,
                    info=f"{scene}:{global_t}", track2d=t2d_src
                )
                W, H = rgb_pil.size

            rgb_list.append(rgb_pil)
            depth_list.append(depth.astype(np.float32))
            K_list.append(K_out.astype(np.float32))
            c2w_list.append(c2w.astype(np.float32))
            size_list.append((W, H))

            t2d_list.append(t2d.astype(np.float32))
            t3d_list.append(t3d.astype(np.float32))
            vis_raw_list.append(vis_raw.astype(np.bool_))

        # 动态筛选 + 最少可见帧（基于原始 vis）
        traj3d_TN3 = np.stack(t3d_list, axis=0).astype(np.float32)
        vis_T_M_raw = np.stack(vis_raw_list, axis=0).astype(bool)

        dyn_mask = self._dynamic_filter(traj3d_TN3, self.dynamic_filter_threshold, self.keep_static_prob, rng)
        if dyn_mask is not None:
            t2d_list = [t2d[dyn_mask] for t2d in t2d_list]
            t3d_list = [t3d[dyn_mask] for t3d in t3d_list]
            vis_T_M_raw = vis_T_M_raw[:, dyn_mask]

        keep_cols = vis_T_M_raw.sum(axis=0) >= self.min_visible_frames
        if keep_cols.any():
            t2d_list = [t2d[keep_cols] for t2d in t2d_list]
            t3d_list = [t3d[keep_cols] for t3d in t3d_list]
            vis_T_M_raw = vis_T_M_raw[:, keep_cols]
        else:
            t2d_list = [np.zeros((0, 2), np.float32) for _ in t2d_list]
            t3d_list = [np.zeros((0, 3), np.float32) for _ in t3d_list]
            vis_T_M_raw = np.zeros((len(vis_raw_list), 0), dtype=bool)

        # 下采样到 target_num
        F_sel, M0 = vis_T_M_raw.shape
        if F_sel > 0 and M0 > 0:
            first_vis = np.where(vis_T_M_raw[0])[0]
            mid_vis = np.where(vis_T_M_raw[F_sel // 2])[0]
            pool = np.unique(np.concatenate([first_vis, mid_vis])) if (first_vis.size + mid_vis.size) > 0 else np.arange(M0, dtype=np.int64)
        else:
            pool = np.array([], dtype=np.int64)

        target = self.target_num
        if pool.size >= target:
            pool = rng.choice(pool, size=target, replace=False)
            t2d_list = [t2d[pool] for t2d in t2d_list]
            t3d_list = [t3d[pool] for t3d in t3d_list]
            vis_T_M_raw = vis_T_M_raw[:, pool]
        else:
            if pool.size == 0:
                M = target
                t2d_list = [np.zeros((M, 2), np.float32) for _ in t2d_list]
                t3d_list = [np.zeros((M, 3), np.float32) for _ in t3d_list]
                vis_T_M_raw = np.zeros((len(vis_raw_list), M), dtype=bool)
            else:
                rep = target // pool.size
                rem = target - rep * pool.size
                pool = np.concatenate([np.tile(pool, rep),
                                       rng.choice(pool, size=rem, replace=True)])
                rng.shuffle(pool)
                t2d_list = [t2d[pool] for t2d in t2d_list]
                t3d_list = [t3d[pool] for t3d in t3d_list]
                vis_T_M_raw = vis_T_M_raw[:, pool]

        # 最终可见性 = 原始 vis ∩ 有限 ∩ 在图内
        track2d_seq_list: List[np.ndarray] = []
        track3d_seq_list: List[np.ndarray] = []
        track_vis_seq_list: List[np.ndarray] = []

        for v_i in range(len(idxs)):
            t2d = t2d_list[v_i]
            t3d = t3d_list[v_i]
            vis_raw = vis_T_M_raw[v_i] if vis_T_M_raw.shape[0] > 0 else np.zeros((t2d.shape[0],), dtype=bool)
            W, H = size_list[v_i]

            finite = np.isfinite(t2d).all(axis=1)
            on_img = (t2d[:, 0] >= 0) & (t2d[:, 0] <= (W - 1)) & (t2d[:, 1] >= 0) & (t2d[:, 1] <= (H - 1))
            vis = vis_raw & finite & on_img

            track2d_seq_list.append(t2d.astype(np.float32))
            track3d_seq_list.append(t3d.astype(np.float32))
            track_vis_seq_list.append(vis.astype(np.bool_))

        views = []
        for v_i, global_t in enumerate(idxs):
            views.append(dict(
                img=rgb_list[v_i],                         # PIL.Image
                depthmap=depth_list[v_i],                 # np.float32 [H,W]
                camera_pose=c2w_list[v_i],                # 4x4
                camera_intrinsics=K_list[v_i],            # 3x3
                dataset=self.dataset_label,
                label=scene,
                instance=str(int(frames[global_t]["frame_number"])),

                track2d=track2d_seq_list[v_i],            # [M,2]
                track3d=track3d_seq_list[v_i],            # [M,3]
                track_visible=track_vis_seq_list[v_i],    # [M]
                frame_indices=np.array(idxs, dtype=np.int32),
            ))
        return views