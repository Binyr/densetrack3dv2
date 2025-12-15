import os
import cv2
import av
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from densetrack3d.datasets.utils import DeltaData

INVALID_SEQ_NAMES = set([
    "0dBwfeDJ-Dg_101835169-left_rectified", "0dBwfeDJ-Dg_108508509-left_rectified",
    "MtAG1S2x1ts_529896563-left_rectified", "MtAG1S2x1ts_536569903-left_rectified",
    "hs8wbvQQ5FM_571571572-left_rectified", "hs8wbvQQ5FM_578244912-left_rectified"
])

class SimpleVideoDepthDataset(Dataset):
    """
    Dataset for a directory structured as:

        data_root/
            train/
                xxx1.mp4
                xxx2.mp4
                ...
            test/
                yyy1.mp4
                yyy2.mp4
                ...

    anns.json 形如:
    {
      "train": {
        "-17g_k6OZ9E_15075377-left_rectified": 199,
        "-17g_k6OZ9E_21775544-left_rectified": 199,
        ...
      },
      "test": { ... }
    }

    对于每个 RGB 视频 `<name>.mp4`，假设有对应的
    `<name>_depth_unidepth.mp4` 存在同一目录。

    __getitem__ 返回: (DeltaData(...), True)，其中:
        - sample.video:      FloatTensor [T, 3, H, W], in [0, 1]
        - sample.videodepth: FloatTensor [T, 1, H, W], in meters (approx)
        - sample.seq_name:   str, 文件名不含扩展名
        - sample.dataset_name: "steore4D"
    """

    def __init__(
        self,
        data_root,
        ann_path,
        split="train",
        seq_len=24,
        resize_short_edge=None,
        debug=False,
        extensions=(".mp4", ".MP4"),
        ck=None,
    ):
        super().__init__()
        assert split in ["train", "test"], f"Unknown split: {split}"

        self.data_root = data_root
        self.split = split
        self.seq_len = seq_len
        self.resize_short_edge = resize_short_edge
        self.debug = debug
        self.extensions = extensions

        split_dir = os.path.join(self.data_root, self.split)
        assert os.path.isdir(split_dir), f"{split_dir} does not exist"

        # 读取 anns
        with open(ann_path, "r") as f:
            anns_all = json.load(f)
        anns = anns_all[split]  # dict: name -> num_frames
        self.anns = anns

        # 根据 anns 构造 video_paths（只按 anns 的 name 来）
        self.names = sorted(list(self.anns.keys()))
        self.names = [x for x in self.names if x not in INVALID_SEQ_NAMES]
        self.video_paths = [
            os.path.join(split_dir, f"{name}.mp4") for name in self.names
        ]
        if ck is not None:
            if ck == 0:
                self.video_paths = self.video_paths[:33000]
                # p = self.video_paths
                # self.video_paths = [p[x] for x in [18019, 18020, 22496, 22497, ]]
            elif ck == 1:
                self.video_paths = self.video_paths[33000:66000]
                # p = self.video_paths
                # self.video_paths = [p[x] for x in [8133, 8134, 8155, 8156, 23375, 23376]]
            else:
                self.video_paths = self.video_paths[66000:]
                # p = self.video_paths
                # self.video_paths = [p[x] for x in [5098, 5099, 7660, 7661]]

        print(f"Found {len(self.video_paths)} videos in split '{self.split}'")

    def __len__(self):
        if self.debug:
            return min(10, len(self.video_paths))
        return len(self.video_paths)

    # --------------------- 用 PyAV 读指定帧 --------------------- #
    def read_video(self, path, frame_ids, modal="rgb"):
        """
        用 PyAV 从 path 中读取指定 frame_ids 的帧。
        modal="rgb"  -> 输出 uint8 RGB24 [N, H, W, 3]
        modal="depth"-> 输出 uint16 RGB48 [N, H, W, 3]
                         这里假设 3 个通道都是同一个 depth
        """
        # 规范化 frame_ids
        frame_ids = [int(i) for i in frame_ids]
        

        frame_ids_sorted = sorted(set(frame_ids))
        wanted_set = set(frame_ids_sorted)
        max_id = frame_ids_sorted[-1]

        frames_dict = {}

        with av.open(path) as container:
            video_stream = container.streams.video[0]

            for fid, frame in enumerate(container.decode(video_stream)):
                if fid > max_id:
                    break
                if fid not in wanted_set:
                    continue

                if modal == "rgb":
                    rgb_frame = frame.reformat(format="rgb24")    # 8bit RGB
                else:
                    rgb_frame = frame.reformat(format="rgb48le")  # 16bit RGB

                img = rgb_frame.to_ndarray()  # (H, W, 3)
                frames_dict[fid] = img

        # 检查有没有缺帧
        missing = [i for i in frame_ids if i not in frames_dict]
        if len(missing) > 0:
            raise RuntimeError(
                f"{path}: some requested frames not decoded: {missing}"
            )

        # 按原始 frame_ids 顺序 stacking（支持重复）
        ordered_frames = [frames_dict[i] for i in frame_ids]
        frames_array = np.stack(ordered_frames, axis=0)  # [N, H, W, 3]

        return frames_array

    def _depth_path_from_rgb_path(self, rgb_path: str) -> str:
        """
        rgb_path: /.../<name>.mp4
        depth_path: /.../<name>_depth_unidepth.mp4
        """
        base, ext = os.path.splitext(rgb_path)
        return base + "_depth_unidepth" + ext

    def __getitem__(self, index):
        """
        只用 PyAV + anns 里的帧数来采样 frame_ids，不再读整段视频。
        """
        got_it = False
        trial = 0
        n = len(self)
        orig_index = index
        while not got_it and trial < n:
            path = self.video_paths[index]
            data = self._load_single_video(path)
            if data is not None:
                return data
            index = (index + 1) % n
            trial += 1

        raise RuntimeError(
            f"Failed to read any video in the dataset (starting from index {orig_index})."
        )

    def __getitem__2(self, index):
        try:
            path = self.video_paths[index]
            data = self._load_single_video(path)
        except:
            data = (DeltaData(
                seq_name=os.path.splitext(os.path.basename(path))[0]
            ), True)
        return data
    
    def _load_single_video(self, path):
        """
        只根据 anns 的 T_total 和 seq_len 计算 frame_ids，
        再用 read_video 只解码这些帧（RGB + depth），然后做 resize。
        """
        seq_name = os.path.splitext(os.path.basename(path))[0]

        # 从 anns 中拿到视频总帧数
        if seq_name not in self.anns:
            raise KeyError(f"{seq_name} not found in anns")
        T_total = int(self.anns[seq_name])

        if T_total <= 0:
            return None

        # ---------- 1. 根据 T_total & seq_len 计算 frame_ids ---------- #
        if self.seq_len is not None:
            if T_total >= self.seq_len:
                start = np.random.randint(0, T_total - self.seq_len + 1)
                end = start + self.seq_len
                frame_ids = np.arange(start, end, dtype=np.int32)
            else:
                # T_total 太短，就用 linspace 采样并允许重复
                frame_ids = np.linspace(0, T_total - 1, self.seq_len).astype(np.int32)
        else:
            frame_ids = np.arange(T_total, dtype=np.int32)

        # ---------- 2. 用 PyAV 读 RGB 指定帧 ---------- #
        rgb_frames = self.read_video(path, frame_ids, modal="rgb")  # [T_sel, H, W, 3], uint8
        T_sel, H, W, C = rgb_frames.shape

        # ---------- 3. Resize (short edge) for RGB ---------- #
        new_h, new_w = H, W
        if self.resize_short_edge is not None:
            short = min(H, W)
            scale = self.resize_short_edge / float(short)
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))

            # 如果你之前就是强制 512x512，就保持一致
            new_h = 384
            new_w = 512

            resized_rgb = np.zeros((T_sel, new_h, new_w, C), dtype=rgb_frames.dtype)
            for t in range(T_sel):
                resized_rgb[t] = cv2.resize(
                    rgb_frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
            rgb_frames = resized_rgb

        # ---------- 4. RGB -> Tensor [T_sel, 3, H, W] in [0,1] ---------- #
        video_rgb = (
            torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0
        )  # [T_sel,3,new_h,new_w]

        # ---------- 5. 用同样的 frame_ids 读 depth 视频 ---------- #
        depth_path = self._depth_path_from_rgb_path(path)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(
                f"Depth video not found for {path}, expected {depth_path}"
            )

        depth_frames = self.read_video(
            depth_path, frame_ids, modal="depth"
        )  # [T_sel, H_d, W_d, 3], uint16

        # 取第一通道作为 depth
        if depth_frames.ndim == 4 and depth_frames.shape[-1] == 3:
            depth_u16 = depth_frames[..., 0]  # [T_sel, H_d, W_d]
        else:
            depth_u16 = depth_frames  # fallback

        # 写的时候是 depth(m) * 1000 -> uint16，这里除以 1000 还原成米（近似）
        depth_m = depth_u16.astype(np.float32) / 1000.0  # [T_sel, H_d, W_d]

        # ---------- 6. 对 depth 做 resize 到和 RGB 一样大小 ---------- #
        if self.resize_short_edge is not None:
            depth_resized = np.zeros((T_sel, new_h, new_w), dtype=np.float32)
            for t in range(T_sel):
                depth_resized[t] = cv2.resize(
                    depth_m[t],
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth_m = depth_resized

        # [T_sel, 1, H, W]
        videodepth = torch.from_numpy(depth_m).unsqueeze(1)

        # ---------- 7. 打包 DeltaData ---------- #
        sample = DeltaData(
            video=video_rgb,         # [T_sel,3,H,W]
            videodepth=videodepth,   # [T_sel,1,H,W]
            depth_init=videodepth[0],
            depth_init_last=videodepth[-1],
            seq_name=seq_name,
            dataset_name="stereo4d",
        )

        return sample, True


# --------------------- Example usage --------------------- #
if __name__ == "__main__":
    root = "/path/to/your/dataset"       # contains train/ and test/
    ann_path = "/path/to/anns.json"      # 上面格式的 json

    dataset = SimpleVideoDepthDataset(
        data_root=root,
        ann_path=ann_path,
        split="train",
        seq_len=24,
        resize_short_edge=512,
        debug=True,   # 只跑前几个
    )

    for i in range(len(dataset)):
        sample, _ = dataset[i]
        print(
            "idx:", i,
            "seq_name:", sample.seq_name,
            "video:", sample.video.shape,
            "depth:", sample.videodepth.shape,
        )
        break
