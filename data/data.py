import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from torchvision import transforms
from PIL import Image


class RubbishDataset(Dataset):
    def __init__(self, mode="train", base="./resources/dataset/", transform=transforms.Compose(transforms.ToTensor())):
        with open(os.path.join(base, mode + "_.json")) as f:
            self.data_dir = json.load(f)
        self.base_dir = base
        self.transform = transform

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, item):
        dic = self.data_dir[item]
        img_dir = os.path.join(self.base_dir, dic["dir"].replace("\\", "/"))
        y = int(dic["label"])
        img = Image.open(img_dir)
        x = self.transform(img)
        return x, y


class RubbishTestSet(Dataset):
    def __init__(self, path="./resources/dataset/test_set", transform=transforms.Compose(transforms.ToTensor())):
        self.path = path
        self.images = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        img_dir = os.path.join(self.path, name)
        img = Image.open(img_dir)
        x = self.transform(img)
        return x, name


def get_rubbish_classification_loader(
    mode="train",
    batch_size=128,
    num_workers=8,
    shuffle=False,
    ddp=False
):
    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5434, 0.5218, 0.4918], std=[0.2796, 0.2745, 0.2848]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5434, 0.5218, 0.4918], std=[0.2796, 0.2745, 0.2848]),
            ]
        )

    dataset = RubbishDataset(mode=mode, transform=transform) if mode != "test" else RubbishTestSet(transform=transform)

    if ddp:
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=ddp_sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader
