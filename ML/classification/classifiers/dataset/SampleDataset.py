import monai
import random
import torch
from monai.data import Dataset, CacheDataset

class SampleDataset(Dataset):
    def __init__(self, data_list, num_frames=10, transforms=None, agg=None):
        super().__init__(data_list, transform=transforms)
        self.num_frames = num_frames
        self.agg = agg

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        image = sample["image"]
        label = sample["label"]
        
        sampled_indices = torch.tensor(sorted(random.sample(range(image.shape[1]), self.num_frames)), dtype=torch.long)
        image_sampled = torch.index_select(image, dim=1, index=sampled_indices)
        if self.agg == "mean":
            image_sampled = torch.mean(image_sampled, dim=1)

        return {"image": image_sampled, "label": label}


class SampleDataset2(Dataset):
    def __init__(self, data_list, num_frames=10, transforms=None, agg=None):
        super().__init__(data_list, transform=transforms)
        self.num_frames = num_frames
        self.agg = agg

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        image = sample["image"]  # Shape: (C, T, H, W)
        label = sample["label"]

        C, T, H, W = image.shape
        num_groups = T // self.num_frames  # Number of new depth slices

        remainder = T % self.num_frames  # Remaining frames

        if remainder > 0:
            image = torch.cat([image, image[:, -1:].repeat(1, self.num_frames - remainder, 1, 1)], dim=1)

        # Reshape consistently
        image_grouped = image[:, :num_groups * self.num_frames].view(C, num_groups, self.num_frames, H, W)
        image_reduced = image_grouped.mean(dim=2)  # Compute mean over num_frames

        return {"image": image_reduced, "label": label}
