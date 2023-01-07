import json
import os
from glob import glob
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import h5py
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from slot_attention.utils import (
    compact,
    rescale,
    slightly_off_center_crop,
    slightly_off_center_mask_crop,
    flatten_all_but_last,
)


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images", split)
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        image_path = self.files[index]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        with open(
            os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")
        ) as f:
            scene = json.load(f)
        paths: List[Optional[str]] = []
        total_num_images = len(scene["scenes"])
        i = 0
        while (
            self.max_num_images is None or len(paths) < self.max_num_images
        ) and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            if num_objects_in_scene <= self.max_n_objects:
                image_path = os.path.join(
                    self.data_path, scene["scenes"][i]["image_filename"]
                )
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
            i += 1
        return sorted(compact(paths))


class CLEVRWithMasksDataset(Dataset):
    # Dataset details: https://github.com/deepmind/multi_object_datasets#clevr-with-masks
    def __init__(
        self,
        data_root: str,
        clevr_transforms: Callable,
        mask_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.mask_transforms = mask_transforms
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"

        self.data = h5py.File(self.data_root, "r")
        if self.max_n_objects:
            if self.split == "train":
                num_objects_in_scene = np.sum(self.data["visibility"][:70_000], axis=1)
            elif self.split == "val":
                num_objects_in_scene = np.sum(
                    self.data["visibility"][70_001:85_000], axis=1
                )
            elif self.split == "test":
                num_objects_in_scene = np.sum(
                    self.data["visibility"][85_001:100_000], axis=1
                )
            else:
                raise NotImplementedError
            self.indices = (
                np.argwhere(num_objects_in_scene <= self.max_n_objects).flatten()
                + {"train": 0, "val": 70_001, "test": 85_001}[self.split]
            )

    def __getitem__(self, index: int):
        if self.max_n_objects:
            index_to_load = self.indices[index]
        else:
            index_to_load = index
        img = self.data["image"][index_to_load]
        if self.split == "train":
            return self.clevr_transforms(img)
        else:
            mask = self.data["mask"][index_to_load]
            return self.clevr_transforms(img), self.mask_transforms(mask)

    def __len__(self):
        return len(self.indices if self.max_n_objects else self.data["image"])


class RAVENSRobotDataset(Dataset):
    # Dataset generated using https://github.com/HHousen/ravens, which is a
    # fork of https://github.com/google-research/ravens.
    def __init__(
        self,
        data_root: str,
        ravens_transforms: Callable,
        mask_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
        only_orientation: Optional[int] = None,  # currently, needs `max_n_objects` set
    ):
        super().__init__()
        self.data_root = data_root
        self.ravens_transforms = ravens_transforms
        self.mask_transforms = mask_transforms
        self.max_n_objects = max_n_objects
        self.split = split
        self.only_orientation = only_orientation
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "test"

        data_path = glob(os.path.join(self.data_root, f"ravens_*_{self.split}.h5"))[0]
        self.data = h5py.File(data_path, "r")
        if self.max_n_objects:
            # `num_objects_on_table` stores the number of objects on the table
            # for a set of 3 angles (images) of one scene. So, expand it to
            # match the size of the images and masks.
            num_objects_on_table = np.repeat(self.data["num_objects_on_table"], 3)
            self.indices = np.argwhere(
                num_objects_on_table <= self.max_n_objects
            ).flatten()
            # Every third image has the same orientation. Thus, changing the
            # start index and getting every third image will retrieve all of
            # one orientation.
            if self.only_orientation is not None:
                self.indices = self.indices[self.only_orientation :: 3]

    def __getitem__(self, index: int):
        if self.max_n_objects:
            index_to_load = self.indices[index]
        else:
            index_to_load = index
        img = self.data["color"][index_to_load]
        if self.split == "train":
            return self.ravens_transforms(img)
        else:
            mask = self.data["segm"][index_to_load]
            return self.ravens_transforms(img), self.mask_transforms(mask)

    def __len__(self):
        return len(self.indices if self.max_n_objects else self.data["color"])


class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        max_n_objects: int,
        num_workers: int,
        resolution: Tuple[int, int],
        clevr_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        with_masks: Optional[bool] = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.mask_transforms = mask_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.resolution = resolution
        self.with_masks = with_masks
        self.max_num_entries = 11

        print(
            f"INFO: limiting the dataset to only images with `max_n_objects` ({max_n_objects}) objects."
        )

        if not self.clevr_transforms:
            # Slot Attention [0] uses the same transforms as IODINE [1]. IODINE
            # claims to do a center crop, but assumes the images have a height
            # of 250, while they actually have a height of 240. We use the same
            # transforms, evevn though they are slightly off.
            # [0]: https://github.com/google-research/google-research/blob/master/slot_attention/data.py#L28
            # [1]: https://github.com/deepmind/deepmind-research/blob/11c2ab53e8afd24afa8904f22fd81b699bfbce6e/iodine/modules/data.py#L191
            # In original tfrecords format, CLEVR (with masks) image shape is
            # (height, width, channels) = (240, 320, 3).
            self.clevr_transforms = transforms.Compose(
                [
                    # image has shape (H x W x C)
                    transforms.ToTensor(),  # rescales to range [0.0, 1.0]
                    # image has shape (C x H x W)
                    transforms.Lambda(rescale),  # rescale between -1 and 1
                    transforms.Lambda(slightly_off_center_crop),
                    transforms.Resize(self.resolution),
                ]
            )

        if not self.mask_transforms:

            def mask_transforms(mask):
                # Based on https://github.com/deepmind/deepmind-research/blob/master/iodine/modules/data.py#L115
                # `mask` has shape [max_num_entities, height, width, channels]
                mask = torch.from_numpy(mask)
                mask = slightly_off_center_mask_crop(mask)
                mask = torch.permute(mask, [0, 3, 1, 2])
                # `mask` has shape [max_num_entities, channels, height, width]
                flat_mask, unflatten = flatten_all_but_last(mask, n_dims=3)
                resize = transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.NEAREST
                )
                flat_mask = resize.forward(flat_mask)
                mask = unflatten(flat_mask)
                # `mask` has shape [max_num_entities, channels, height, width]
                mask = torch.permute(mask, [0, 2, 3, 1])
                # `mask` has shape [max_num_entities, height, width, channels]
                return mask

            self.mask_transforms = mask_transforms

        dataset_object = CLEVRWithMasksDataset if self.with_masks else CLEVRDataset
        self.train_dataset = dataset_object(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
            mask_transforms=self.mask_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
        )
        self.val_dataset = dataset_object(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
            mask_transforms=self.mask_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class RAVENSRobotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        max_n_objects: int,
        num_workers: int,
        resolution: Tuple[int, int],
        ravens_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        alternative_crop: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.ravens_transforms = ravens_transforms
        self.mask_transforms = mask_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.resolution = resolution
        self.alternative_crop = alternative_crop
        # There will be a maximum of 12 items in the segmentation mask: 8 items
        # on the table plus the background, table, robot, and robot arm.
        self.max_num_entries = 12

        print(
            f"INFO: limiting the dataset to only images with `max_n_objects` ({max_n_objects}) objects."
        )

        alt_crop_func = lambda img: img[:, 232:-10, 50:-50]
        if not self.ravens_transforms:
            self.ravens_transforms = transforms.Compose(
                [
                    # image has shape (H x W x C)
                    transforms.ToTensor(),  # rescales to range [0.0, 1.0]
                    # image has shape (C x H x W)
                    transforms.Lambda(rescale),  # rescale between -1 and 1
                    transforms.Lambda(alt_crop_func)
                    if self.alternative_crop
                    else transforms.CenterCrop((480, 500)),
                    transforms.Resize(self.resolution),
                ]
            )

        if not self.mask_transforms:

            def mask_transforms(mask):
                # `mask` has shape [height, width]
                mask = torch.from_numpy(mask)
                mask = mask.unsqueeze(0)
                # `mask` has shape [channel, height, width]
                transform_func = transforms.Compose(
                    [
                        transforms.Lambda(alt_crop_func)
                        if self.alternative_crop
                        else transforms.CenterCrop((480, 500)),
                        transforms.Resize(
                            self.resolution,
                            interpolation=transforms.InterpolationMode.NEAREST,
                        ),
                    ]
                )
                mask = transform_func(mask)
                mask = mask.permute([1, 2, 0])
                # `mask` has shape [height, width, channel]
                mask = F.one_hot(mask.to(torch.int64), self.max_num_entries)
                # `mask` has shape [height, width, channel, max_num_entries]
                mask = mask.permute([3, 0, 1, 2])
                # `mask` has shape [max_num_entries, height, width, channel]
                return mask

            self.mask_transforms = mask_transforms

        self.train_dataset = RAVENSRobotDataset(
            data_root=self.data_root,
            ravens_transforms=self.ravens_transforms,
            mask_transforms=self.mask_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
            only_orientation=0 if self.alternative_crop else None,
        )
        self.val_dataset = RAVENSRobotDataset(
            data_root=self.data_root,
            ravens_transforms=self.ravens_transforms,
            mask_transforms=self.mask_transforms,
            split="test",
            max_n_objects=self.max_n_objects,
            only_orientation=0 if self.alternative_crop else None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class Shapes3D(Dataset):
    # From https://github.com/singhgautam/slate/blob/6afe75211a79ef7327071ce198f4427928418bf5/shapes_3d.py
    # Download from https://console.cloud.google.com/storage/browser/3d-shapes (https://storage.googleapis.com/3d-shapes/3dshapes.h5)
    def __init__(self, root, phase):
        assert phase in ["train", "val", "test"]
        with h5py.File(root, "r") as f:
            if phase == "train":
                self.imgs = f["images"][:400000]
            elif phase == "val":
                self.imgs = f["images"][400001:430000]
            elif phase == "test":
                self.imgs = f["images"][430001:460000]
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        img = self.imgs[index]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255.0

        return img

    def __len__(self):
        return len(self.imgs)


class Shapes3dDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.train_dataset = Shapes3D(root=self.data_root, phase="train",)
        self.val_dataset = Shapes3D(root=self.data_root, phase="val",)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
