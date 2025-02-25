import monai
import random
import torch
from monai.data import Dataset, CacheDataset

class ClassificationDataset():
    def __init__(self, data_list, start_frame: int, end_frame: int, agg: str, cache: bool, transforms=None):

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.agg = agg

        if cache:
            self.dataset = CacheDataset(data=data_list, transform=transforms)
        else:
            self.dataset = Dataset(data=data_list, transform=transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]
        
        total_frames = image.shape[1]

        end_frame = self.end_frame if self.end_frame is not None else total_frames
        start_frame = torch.clamp(torch.tensor(self.start_frame), 0, total_frames - 1).item()
        end_frame = torch.clamp(torch.tensor(self.end_frame if self.end_frame else total_frames), start_frame + 1, total_frames).item()

        image = image[:, start_frame:end_frame, :, :]

        image = image.mean(dim=1, keepdim=False) if self.agg == "mean" else image

        return {"image": image, "label": label}