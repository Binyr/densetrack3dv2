import glob
from pathlib import Path
import os
import numpy as np
import cv2
import torch
from PIL import Image

from densetrack3d.datasets.kubric_dataset import BasicDataset
from densetrack3d.datasets.utils import DeltaData, add_noise_depth, aug_depth


class PointOdysseyDataset(BasicDataset):
    """
    data_root:  指向 /mnt/shared-storage-user/dongjunting-group/DATA/point0dyssey
    split:      'train' / 'val' / 'test'
    其余参数沿用 BasicDataset，比如 seq_len, traj_per_sample, crop_size 等。
    """
    def __init__(
        self,
        data_root,
        split="train",
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        # BasicDataset 里会记录 self.data_root 等
        super().__init__(
            data_root=os.path.join(data_root, split),
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )
        self.split = split

        # 找到所有有 anno.npz 的 sequence 目录
        self.scenes = []
        self.sample_index = []   # 每个元素是 (scene_idx, start_frame)

        root = Path(self.data_root)
        for scene_dir in sorted(root.iterdir()):
            if not scene_dir.is_dir():
                continue

            anno_path = scene_dir / "anno.npz"
            rgb_dir = scene_dir / "rgbs"
            depth_dir = scene_dir / "depths"

            if not anno_path.is_file():
                continue
            if not rgb_dir.is_dir() or not depth_dir.is_dir():
                continue

            # 这里只读一下 shape，不把整个 npz 常驻内存
            anno = np.load(str(anno_path))
            T = int(anno["trajs_2d"].shape[0])
            anno.close()

            # 对应的 rgb_xxxxx.jpg / depth_xxxxx.png
            rgb_files = sorted(glob.glob(str(rgb_dir / "rgb_*.jpg")))
            depth_files = sorted(glob.glob(str(depth_dir / "depth_*.png")))

            if len(rgb_files) < T or len(depth_files) < T:
                print(f"[PointOdysseyDataset] skip {scene_dir}, frames in anno={T}, "
                      f"rgb={len(rgb_files)}, depth={len(depth_files)}")
                continue

            scene_id = len(self.scenes)
            self.scenes.append(
                dict(
                    name=scene_dir.name,
                    root=str(scene_dir),
                    anno_path=str(anno_path),
                    T=T,
                    rgb_files=rgb_files,
                    depth_files=depth_files,
                )
            )

            # 为这个 scene 预先生成若干个 (scene_idx, start) 样本
            if T >= self.seq_len:
                # 步长可以调节；这里用不重叠的窗，方便控制 __len__
                for start in range(0, T - self.seq_len + 1, self.seq_len):
                    self.sample_index.append((scene_id, start))

        if len(self.sample_index) == 0:
            raise RuntimeError("PointOdysseyDataset: no valid sequences found under "
                               f"{self.data_root}")

        print(f"[PointOdysseyDataset] split={split}, scenes={len(self.scenes)}, "
              f"samples={len(self.sample_index)}")

    def __len__(self):
        return len(self.sample_index)

    def getitem_helper(self, index):
        """
        BasicDataset.__getitem__ 会不断调用这个函数直到 gotit=True。
        在这里完成：
        1) 选择一个 sequence 的时间窗
        2) 读取 RGB / depth
        3) 从 anno.npz 中抽取对应时间窗和轨迹子集
        4) 转成 torch tensor，构造 DeltaData（或者你自己的 sample 对象）
        """
        try:
            scene_idx, start = self.sample_index[index % len(self.sample_index)]
        except Exception as e:
            print(f"[PointOdysseyDataset] index error: {e}")
            return None, False

        scene = self.scenes[scene_idx]
        T = scene["T"]

        if start + self.seq_len > T:
            # 理论上不应该发生，因为构建 sample_index 时已经检查过
            return None, False

        # 读取 anno.npz 中的轨迹和相机参数
        anno = np.load(scene["anno_path"])
        trajs_2d_all = anno["trajs_2d"]      # (T, N, 2)
        trajs_3d_all = anno["trajs_3d"]      # (T, N, 3)
        visibs_all   = anno["visibs"]        # (T, N)
        valids_all   = anno["valids"]        # (T, N)
        intr_all     = anno["intrinsics"]    # (T, 3, 3)
        extr_all     = anno["extrinsics"]    # (T, 4, 4)

        num_traj = trajs_2d_all.shape[1]
        if num_traj < self.traj_per_sample:
            # 这个 sequence 轨迹太少，跳过
            anno.close()
            return None, False

        # 随机抽取一部分轨迹
        traj_ids = np.random.choice(num_traj, self.traj_per_sample, replace=False)

        t0 = start
        t1 = start + self.seq_len

        trajs_2d = trajs_2d_all[t0:t1, traj_ids]         # (seq_len, traj_per_sample, 2)
        trajs_3d = trajs_3d_all[t0:t1, traj_ids]         # (seq_len, traj_per_sample, 3)
        visibs   = visibs_all[t0:t1, traj_ids]           # (seq_len, traj_per_sample)
        valids   = valids_all[t0:t1, traj_ids]           # (seq_len, traj_per_sample)
        intr     = intr_all[t0:t1]                       # (seq_len, 3, 3)
        extr     = extr_all[t0:t1]                       # (seq_len, 4, 4)
        anno.close()

        # 读取这一段时间窗的 RGB 和 depth
        rgb_list = []
        depth_list = []
        for t in range(t0, t1):
            rgb_path = scene["rgb_files"][t]
            depth_path = scene["depth_files"][t]

            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                print(f"[PointOdysseyDataset] fail to read {rgb_path}")
                return None, False
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"[PointOdysseyDataset] fail to read {depth_path}")
                return None, False

            rgb_list.append(rgb)
            depth_list.append(depth)

        rgb_np = np.stack(rgb_list, axis=0)      # (T, H, W, 3)
        depth_np = np.stack(depth_list, axis=0)  # (T, H, W) 或 (T, H, W, 1)

        # 这里可以按你自己想要的方式做 resize / crop / flip
        # 为了简单，这里不做随机 resize，只做中心 crop 到 crop_size
        T_seq, H, W, _ = rgb_np.shape
        crop_h, crop_w = self.crop_size

        # 如果比 crop_size 小，就简单 pad；比 crop_size 大就中心裁剪
        pad_h = max(0, crop_h - H)
        pad_w = max(0, crop_w - W)
        if pad_h > 0 or pad_w > 0:
            rgb_np = np.pad(
                rgb_np,
                ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            depth_np = np.pad(
                depth_np,
                ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2)),
                mode="constant",
                constant_values=0,
            )
            H, W = rgb_np.shape[1], rgb_np.shape[2]

        if H > crop_h or W > crop_w:
            off_h = (H - crop_h) // 2
            off_w = (W - crop_w) // 2
            rgb_np = rgb_np[:, off_h:off_h + crop_h, off_w:off_w + crop_w, :]
            depth_np = depth_np[:, off_h:off_h + crop_h, off_w:off_w + crop_w]
            # 轨迹坐标需要减去偏移
            trajs_2d[..., 0] -= off_w
            trajs_2d[..., 1] -= off_h

        # 转成 torch tensor，归一化到 [0,1] 或 [-1,1] 看你模型那边怎么用
        video = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        depth = torch.from_numpy(depth_np).unsqueeze(1).float()  # (T, 1, H, W)

        trajs_2d = torch.from_numpy(trajs_2d).float()
        trajs_3d = torch.from_numpy(trajs_3d).float()
        visibs   = torch.from_numpy(visibs).float()
        valids   = torch.from_numpy(valids).float()
        intr     = torch.from_numpy(intr).float()
        extr     = torch.from_numpy(extr).float()

        # 如果需要 photometric augmentation，可以在这里调用 BasicDataset 里定义的函数，
        # 不过你要看一下 add_photometric_augs 的具体实现，需要哪些字段。
        # 例如，只对视频做颜色增广：
        if self.use_augs:
            # 简单示例：逐帧调用 ColorJitter / GaussianBlur
            aug_video = []
            for t in range(video.shape[0]):
                img = video[t]  # (3,H,W)
                img_pil = Image.fromarray(
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                if np.random.rand() < self.color_aug_prob:
                    img_pil = self.photo_aug(img_pil)
                if np.random.rand() < self.blur_aug_prob:
                    img_pil = self.blur_aug(img_pil)
                img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
                aug_video.append(img_tensor)
            video = torch.stack(aug_video, dim=0)

        # 这里的 DeltaData 字段名你需要对照 densetrack3d/datasets/utils.py 里的定义改一下
        # 我先写一个比较自然的版本：
        sample = DeltaData(
            video=video,                 # (seq_len, 3, H, W)
            videodepth=depth,                 # (seq_len, 1, H, W)
            trajs_2d=trajs_2d,           # (seq_len, traj_per_sample, 2)
            trajs_3d=trajs_3d,           # (seq_len, traj_per_sample, 3)
            visibs=visibs,               # (seq_len, traj_per_sample)
            valids=valids,               # (seq_len, traj_per_sample)
            intrinsics=intr,             # (seq_len, 3, 3)
            extrinsics=extr,             # (seq_len, 4, 4)
        )

        return sample, True