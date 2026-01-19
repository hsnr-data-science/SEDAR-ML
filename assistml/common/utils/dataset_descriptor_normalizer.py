from typing import Dict, Optional

import numpy as np
from beanie import PydanticObjectId

from common.data import Dataset


class DatasetDescriptorNormalizer:

    _descriptors: Dict[PydanticObjectId, np.ndarray]
    _min_vector: Optional[np.ndarray]
    _max_vector: Optional[np.ndarray]
    _range_vector: Optional[np.ndarray]

    def __init__(self):
        self._descriptors = {}
        self._min_vector = None
        self._max_vector = None
        self._range_vector = None

    def add_dataset(self, dataset: Dataset):
        if dataset.id not in self._descriptors:
            self._descriptors[dataset.id] = dataset.get_dataset_descriptor()

    def fit_normalizers(self) -> None:
        descriptors = list(self._descriptors.values())
        self._min_vector = np.min(descriptors, axis=0)
        self._max_vector = np.max(descriptors, axis=0)

        # Avoid division by zero
        self._range_vector = self._max_vector - self._min_vector
        self._range_vector[self._range_vector == 0] = 1

    def normalize(self, descriptor: np.ndarray) -> np.ndarray:
        if self._min_vector is None or self._max_vector is None or self._range_vector is None:
            raise ValueError("Normalizers not fitted")
        return (descriptor - self._min_vector) / self._range_vector
