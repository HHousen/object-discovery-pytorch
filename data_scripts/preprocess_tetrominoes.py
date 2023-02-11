# Script to convert Tetrominoes tfrecords file to h5py.
# Tetrominoes information: https://github.com/deepmind/multi_object_datasets#tetrominoes
# Download: https://console.cloud.google.com/storage/browser/multi-object-datasets/tetrominoes

import numpy as np
from tqdm import tqdm
import tetrominoes
import h5py


dataset = tetrominoes.dataset("tetrominoes_train.tfrecords").as_numpy_iterator()

with h5py.File("tetrominoes.h5", "w") as f:
    for idx, entry in tqdm(enumerate(dataset), total=1_000_000):
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
