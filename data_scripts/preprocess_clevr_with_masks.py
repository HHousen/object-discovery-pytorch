# Script to convert CLEVR (with masks) tfrecords file to h5py.
# CLEVR (with masks) information: https://github.com/deepmind/multi_object_datasets#clevr-with-masks
# Download: https://console.cloud.google.com/storage/browser/multi-object-datasets/clevr_with_masks

import numpy as np
from tqdm import tqdm
import clevr_with_masks
import h5py


dataset = clevr_with_masks.dataset(
    "clevr_with_masks_train.tfrecords"
).as_numpy_iterator()

with h5py.File("data.h5", "w") as f:
    for idx, entry in tqdm(enumerate(dataset), total=100_000):
        for key, value in entry.items():
            value = value[np.newaxis, ...]
            if idx == 0:
                f.create_dataset(
                    key,
                    data=value,
                    dtype=value.dtype,
                    maxshape=(None, *value.shape[1:]),
                    compression="gzip",
                    chunks=True,
                )
            else:
                f[key].resize((f[key].shape[0] + value.shape[0]), axis=0)
                f[key][-value.shape[0] :] = value
