import torch
from densetrack3d.datasets.tapvid3d_dataset2 import TapVid3DDataset

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, repeats=1):
        if isinstance(repeats, int):
            repeats = [repeats] * len(dataset_list)

        num_objects = 0
        for dataset, repeat in zip(dataset_list, repeats):
            num_objects += len(dataset) * repeat
        
        # global idx
        global_idxes = [x for x in range(num_objects)]
        # local dataset and its idx
        pairs = []
        for did, dataset in enumerate(dataset_list):
            for _ in range(repeats[did]):
                pairs.extend([(did, i) for i in range(len(dataset))])
        
        # mapping
        global_to_local = {}
        for gid, l_pair in zip(global_idxes, pairs):
            global_to_local[gid] = l_pair
        
        self.global_to_local = global_to_local
        self.dataset_list = dataset_list

        print(f"merge {len(dataset_list)} dataset, total {num_objects} samples")
    
    def __len__(self, ):
        return len(self.global_to_local)

    def __getitem__(self, idx):
        did, lid = self.global_to_local[idx]
        if isinstance(self.dataset_list[did], TapVid3DDataset):
            return self.dataset_list[did].__getitem__(lid), True
        else:
            return self.dataset_list[did].__getitem__(lid)
    
    def worker_init_fn(self, worker_id):
        # print(worker_id)
        for i, dataset in enumerate(self.dataset_list):
            if hasattr(dataset, "set_seed"):
                dataset.set_seed(worker_id*10 + i*100)
        self.worker_id = worker_id