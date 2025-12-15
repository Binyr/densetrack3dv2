import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from densetrack3d.datasets.utils import DeltaData

class SimpleVideoDataset(Dataset):
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

    Returns a dict with:
        video: FloatTensor [T, 3, H, W], in [0, 1]
        seq_name: str, video file name without extension
        path: str, full path to the video file
    """

    def __init__(
        self,
        data_root,
        split="train",
        seq_len=24,
        resize_short_edge=None,
        debug=False,
        extensions=(".mp4", ".MP4"),
        chunck=0,
    ):
        """
        Args:
            data_root (str): Root folder containing 'train' and 'test'.
            split (str): 'train' or 'test'.
            seq_len (int or None): If not None, sample a clip of length seq_len.
            resize_short_edge (int or None): If set, resize frames so that
                the shorter side == resize_short_edge (keeping aspect ratio).
            debug (bool): If True, limit dataset length (e.g. 10).
            extensions (tuple): Video file extensions to consider.
        """
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

        self.video_paths = sorted(
            [
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(self.extensions) and "depth" not in f
            ]
        )
        if True:
            self.video_paths = ["datasets/stereo4d/huggingface/train/zzW5NZGERmU_294060727-left_rectified.mp4", "datasets/stereo4d/huggingface/train/-17g_k6OZ9E_15075377-left_rectified.mp4"]
            self.video_paths = ["datasets/stereo4d/huggingface/train/hs8wbvQQ5FM_571571572-left_rectified.mp4"]
        else:
            if chunck == 0:
                self.video_paths = self.video_paths[:33000]
            elif chunck == 1:
                self.video_paths = self.video_paths[33000:66000]
            else:
                self.video_paths = self.video_paths[66000:]
            if len(self.video_paths) == 0:
                raise RuntimeError(f"No video files found in {split_dir}")

        print(f"Found {len(self.video_paths)} videos in split '{self.split}'")

    def __len__(self):
        if self.debug:
            return min(10, len(self.video_paths))
        return len(self.video_paths)

    def __getitem__(self, index):
        """
        Robust __getitem__ that skips unreadable videos.
        """
        got_it = False
        trial = 0
        n = len(self)
        while not got_it and trial < n:
            path = self.video_paths[index]
            data = self._load_single_video(path)
            if data is not None:
                return data
            # If failed, move to next index
            index = (index + 1) % n
            trial += 1

        # If all failed, raise error
        raise RuntimeError("Failed to read any video in the dataset.")

    def _load_single_video(self, path):
        """
        Load a single .mp4 as [T, 3, H, W] tensor (float, [0,1]).
        Optionally resize and sample a subsequence.
        """
        # ---- 1. Read all frames with OpenCV ----
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            # unreadable / empty video
            return None

        video = np.stack(frames, axis=0)  # [T, H, W, 3]

        # ---- 2. Optional resize (short edge) ----
        if self.resize_short_edge is not None:
            T, H, W, C = video.shape
            short = min(H, W)
            scale = self.resize_short_edge / float(short)
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
            new_h = 512
            new_w = 512

            resized = np.zeros((T, new_h, new_w, C), dtype=video.dtype)
            for t in range(T):
                resized[t] = cv2.resize(video[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            video = resized

        # ---- 3. Clip sampling (seq_len) ----
        T = video.shape[0]
        if self.seq_len is not None:
            if T >= self.seq_len:
                # sample a random contiguous subsequence
                start = np.random.randint(0, T - self.seq_len + 1)
                end = start + self.seq_len
                video = video[start:end]
            else:
                # if too short, sample with repetition using linspace indices
                idx = np.linspace(0, T - 1, self.seq_len).astype(np.int32)
                video = video[idx]

        # ---- 4. To tensor [T, 3, H, W], float in [0,1] ----
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0

        sample = {
            "video": video,  # [T, 3, H, W]
            "seq_name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "split": self.split,
        }

        sample = DeltaData(
            video=video,
            # videodepth=torch.zeros_like(video),
            # segmentation=segs,
            # trajectory=trajectory_2d,
            # trajectory3d=trajectory_3d,
            # visibility=visibility,
            seq_name=f"{os.path.splitext(os.path.basename(path))[0]}",
            # query_points=None,
            # intrs=intrinsic_mat,
            dataset_name="steore4D",
            # trajectory_d=trajectory_d,
            # depth_init=videodepth[0].clone(),
            # depth_init_last=videodepth[-1].clone()
        )
        return sample, True


# Example usage
if __name__ == "__main__":
    root = "/path/to/your/dataset"  # contains train/ and test/

    train_set = SimpleVideoDataset(
        data_root=root,
        split="train",
        seq_len=24,
        resize_short_edge=512,
        debug=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for batch in train_loader:
        vids = batch["video"]   # [B, T, 3, H, W]
        names = batch["seq_name"]
        print(vids.shape, names)
        break
