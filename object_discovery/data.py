import json
import os
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from object_discovery.utils import (
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
            vis = self.data["visibility"][index_to_load]
            return self.clevr_transforms(img), self.mask_transforms(mask), vis

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
        self.max_num_entries = 12
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
            num_objects = np.max(mask) + 1
            vis = torch.zeros(self.max_num_entries)
            vis[range(num_objects)] = 1
            return self.ravens_transforms(img), self.mask_transforms(mask), vis

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
        neg_1_to_pos_1_scale: bool = True,
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
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale
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
            current_transforms = [
                # image has shape (H x W x C)
                transforms.ToTensor(),  # rescales to range [0.0, 1.0]
                # image has shape (C x H x W)
            ]
            if self.neg_1_to_pos_1_scale:
                current_transforms.append(
                    transforms.Lambda(rescale)
                )  # rescale between -1 and 1
            current_transforms.extend(
                [
                    transforms.Lambda(slightly_off_center_crop),
                    transforms.Resize(self.resolution),
                ]
            )
            self.clevr_transforms = transforms.Compose(current_transforms)

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
        neg_1_to_pos_1_scale: bool = True,
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
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale
        # There will be a maximum of 12 items in the segmentation mask: 8 items
        # on the table plus the background, table, robot, and robot arm.
        self.max_num_entries = 12

        print(
            f"INFO: limiting the dataset to only images with `max_n_objects` ({max_n_objects}) objects."
        )

        alt_crop_func = lambda img: img[:, 232:-10, 50:-50]
        if not self.ravens_transforms:
            current_transforms = [
                # image has shape (H x W x C)
                transforms.ToTensor(),  # rescales to range [0.0, 1.0]
                # image has shape (C x H x W)
            ]
            if self.neg_1_to_pos_1_scale:
                current_transforms.append(
                    transforms.Lambda(rescale)
                )  # rescale between -1 and 1
            current_transforms.extend(
                [
                    transforms.Lambda(alt_crop_func)
                    if self.alternative_crop
                    else transforms.CenterCrop((480, 500)),
                    transforms.Resize(self.resolution),
                ]
            )
            self.ravens_transforms = transforms.Compose(current_transforms)

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
    def __init__(self, root, phase, neg_1_to_pos_1_scale: bool = False):
        assert phase in ["train", "val", "test"]
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale

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
        if self.neg_1_to_pos_1_scale:
            img = rescale(img)

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
        neg_1_to_pos_1_scale: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale

        self.train_dataset = Shapes3D(
            root=self.data_root,
            phase="train",
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
        )
        self.val_dataset = Shapes3D(
            root=self.data_root, phase="val", neg_1_to_pos_1_scale=neg_1_to_pos_1_scale
        )

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


class SketchyDataset(Dataset):
    def __init__(self, data_dir, mode, neg_1_to_pos_1_scale: bool = True):
        current_transforms = [transforms.ToTensor()]
        if neg_1_to_pos_1_scale:
            current_transforms.append(transforms.Lambda(rescale))
        self.transforms = transforms.Compose(current_transforms)

        split_file = f"{data_dir}/processed/{mode}_images.txt"
        if os.path.exists(split_file):
            print(f"Reading paths for {mode} files...")
            with open(split_file, "r") as f:
                self.filenames = f.readlines()
            self.filenames = [item.strip() for item in self.filenames]
        else:
            print(f"Searching for {mode} files...")
            self.filenames = glob(f"{data_dir}/processed/{mode}/ep*/ep*.png")
            with open(split_file, "w") as f:
                for item in self.filenames:
                    f.write(f"{item}\n")
        print(f"Found {len(self.filenames)}.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        img = Image.open(file)
        return self.transforms(img)


class SketchyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        neg_1_to_pos_1_scale: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale

        self.train_dataset = SketchyDataset(
            data_dir=self.data_root,
            mode="train",
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
        )
        self.val_dataset = SketchyDataset(
            data_dir=self.data_root,
            mode="valid",
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
        )

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


class ClevrTexDataset(Dataset):
    splits = {"test": (0.0, 0.1), "val": (0.1, 0.2), "train": (0.2, 1.0)}
    variants = {"full", "pbg", "vbg", "grassbg", "camo", "outd"}
    max_num_entries = 11

    def _reindex(self):
        print(f"Indexing {self.basepath}")

        img_index = {}
        msk_index = {}
        met_index = {}

        prefix = f"CLEVRTEX_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"
        met_suffix = ".json"

        _max = 0
        for img_path in tqdm(
            self.basepath.glob(f"**/{prefix}??????{img_suffix}"),
            desc="Indexing",
            total=50_000,
        ):
            indstr = img_path.name.replace(prefix, "").replace(img_suffix, "")
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            met_path = img_path.parent / f"{prefix}{indstr}{met_suffix}"
            indstr_stripped = indstr.lstrip("0")
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise ValueError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise ValueError(f"Duplica {ind}")

            img_index[ind] = str(img_path)
            msk_index[ind] = str(msk_path)
            if not met_path.exists():
                raise ValueError(f"Missing {met_path.name}")
            met_index[ind] = str(met_path)

        if len(img_index) == 0:
            raise ValueError(f"No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            raise ValueError(f"Missing images numbers {missing}")

        return img_index, msk_index, met_index

    def _variant_subfolder(self):
        return f"clevrtex_{self.dataset_variant.lower()}"

    def __init__(
        self,
        path,
        dataset_variant="full",
        split="train",
        crop=True,
        resize=(128, 128),
        neg_1_to_pos_1_scale=True,
        max_n_objects: int = 10,
        index_cache_dir: str = "data/cache/",
    ):
        self.crop = crop
        self.resize = resize
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale
        self.max_n_objects = max_n_objects
        self.index_cache_dir = index_cache_dir
        if dataset_variant not in self.variants:
            raise ValueError(
                f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available "
            )

        if split not in self.splits:
            raise ValueError(
                f"Unknown split {split}; [{', '.join(self.splits)}] available "
            )
        if dataset_variant == "outd":
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split

        self.basepath = Path(path)

        if self.index_cache_dir:
            os.makedirs(index_cache_dir, exist_ok=True)
            index_path = os.path.join(
                self.index_cache_dir, f"index_{split}_{max_n_objects}.npy"
            )
            mask_index_path = os.path.join(
                self.index_cache_dir, f"mask_index_{split}_{max_n_objects}.npy"
            )
            if os.path.isfile(index_path) and os.path.isfile(mask_index_path):
                print(f"Loading {index_path} and {mask_index_path}")
                self.index = np.load(index_path)
                self.mask_index = np.load(mask_index_path)
                return

        full_data_path = os.path.join(self.index_cache_dir, f"full_data.npy")
        if self.index_cache_dir and os.path.isfile(full_data_path):
            print(f"Loading {full_data_path}")
            self.index, self.mask_index, metadata_index = np.load(full_data_path)
        else:
            self.index, self.mask_index, metadata_index = self._reindex()

            print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

            print("Converting dataset image paths to numpy array")
            self.index = np.array(list(dict(sorted(self.index.items())).values()))
            self.mask_index = np.array(
                list(dict(sorted(self.mask_index.items())).values())
            )
            metadata_index = np.array(
                list(dict(sorted(metadata_index.items())).values())
            )
            np.save(
                full_data_path, np.array([self.index, self.mask_index, metadata_index])
            )

        bias, limit = self.splits.get(split, (0.0, 1.0))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))

        self.index = self.index[bias:limit]
        self.mask_index = self.mask_index[bias:limit]
        metadata_index = metadata_index[bias:limit]

        if self.max_n_objects:
            idxs_to_remove = []
            for idx, metadata_path in tqdm(
                enumerate(metadata_index),
                desc="Reading metadata",
                total=len(metadata_index),
            ):
                with open(metadata_path, "r") as file:
                    metadata = self._format_metadata(json.load(file))
                num_objects_in_scene = len(metadata["objects"])
                if num_objects_in_scene > self.max_n_objects:
                    idxs_to_remove.append(idx)
            self.index = np.delete(self.index, idxs_to_remove)
            self.mask_index = np.delete(self.mask_index, idxs_to_remove)

        if self.index_cache_dir:
            np.save(index_path, self.index)
            np.save(mask_index_path, self.mask_index)

    def _format_metadata(self, meta):
        """
        Drop unimportanat, unsued or incorrect data from metadata.
        Data may become incorrect due to transformations,
        such as cropping and resizing would make pixel coordinates incorrect.
        Furthermore, only VBG dataset has color assigned to objects, we delete the value for others.
        """
        objs = []
        for obj in meta["objects"]:
            o = {
                "material": obj["material"],
                "shape": obj["shape"],
                "size": obj["size"],
                "rotation": obj["rotation"],
            }
            if self.dataset_variant == "vbg":
                o["color"] = obj["color"]
            objs.append(o)
        return {"ground_material": meta["ground_material"], "objects": objs}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):
        img = Image.open(self.index[ind])
        msk = Image.open(self.mask_index[ind])

        if self.crop:
            crop_size = int(0.8 * float(min(img.width, img.height)))
            img = img.crop(
                (
                    (img.width - crop_size) // 2,
                    (img.height - crop_size) // 2,
                    (img.width + crop_size) // 2,
                    (img.height + crop_size) // 2,
                )
            )
            msk = msk.crop(
                (
                    (msk.width - crop_size) // 2,
                    (msk.height - crop_size) // 2,
                    (msk.width + crop_size) // 2,
                    (msk.height + crop_size) // 2,
                )
            )
        if self.resize:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            msk = msk.resize(self.resize, resample=Image.NEAREST)

        img = transforms.functional.to_tensor(np.array(img)[..., :3])
        if self.neg_1_to_pos_1_scale:
            img = rescale(img)
        msk = torch.from_numpy(np.array(msk))[None]
        # `msk` has shape [channel, height, width]
        msk = msk.permute([1, 2, 0])
        # `msk` has shape [height, width, channel]
        msk = F.one_hot(msk.to(torch.int64), self.max_num_entries)
        # `msk` has shape [height, width, channel, max_num_entries]
        msk = msk.permute([3, 0, 1, 2])
        # `msk` has shape [max_num_entries, height, width, channel]
        if self.split == "train":
            return img
        num_objects = torch.max(msk) + 1
        vis = torch.zeros(self.max_num_entries)
        vis[range(num_objects)] = 1
        return img, msk, vis


class ClevrTexDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        max_n_objects: int = 10,
        resolution: Optional[Tuple[int, int]] = None,
        neg_1_to_pos_1_scale: bool = True,
        dataset_variant: str = "full",
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.max_n_objects = max_n_objects
        self.neg_1_to_pos_1_scale = neg_1_to_pos_1_scale
        self.dataset_variant = dataset_variant
        self.resolution = resolution
        self.max_num_entries = 11

        self.train_dataset = ClevrTexDataset(
            path=self.data_root,
            dataset_variant=self.dataset_variant,
            split="train",
            crop=bool(self.resolution),
            resize=self.resolution if self.resolution else False,
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
            max_n_objects=max_n_objects,
        )
        self.val_dataset = ClevrTexDataset(
            path=self.data_root,
            dataset_variant=self.dataset_variant,
            split="val",
            crop=bool(self.resolution),
            resize=self.resolution if self.resolution else False,
            neg_1_to_pos_1_scale=neg_1_to_pos_1_scale,
            max_n_objects=max_n_objects,
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
