from typing import Optional
from torch.utils.data import Dataset
import collections
from torch.utils.data._utils.collate import default_collate
import numpy as np
import warnings


class DatasetOfSequences(Dataset):
    """
    A Dataset which concatenates data into for a small window of frames

    Takes a list of Datasets, each corresponding to a sequence.
    The dataset_config created returns a Dataset of windows.
    Each window takes consecutive frames of a given sequence and
    concatenates them

    Parameters
        sequence_len (int): The length of a window of frames
        dataset_list (list): The list of dataset_config
        sequence_ids (list): The list ids of the dataset_config list
        sequence_transforms (callable): A Transform to be applied
    """

    def __init__(self,
                 sequence_len: int,
                 dataset_list: list,
                 sequence_ids: list = None,
                 transform: Optional[callable] = None,
                 sequence_transforms: Optional[callable] = None,
                 num_points: int = -1,
                 stride: int = 1):
        # -- ME -- dataset_list: list of `Sequences`
        # A `Sequence` is a `Dataset`.
        # In case of Kitti360, a `Sequence.__get__` returns a data_dict that represents a frame
        self.datasets: list = dataset_list
        self.dataset_sizes: list = [0]
        self.sequence_len: int = sequence_len
        # -- ME -- in Kitti360, `transform` is identity
        self.transform: Optional[callable] = transform
        self.sequence_transforms: Optional[callable] = sequence_transforms
        self.stride = stride
        self.num_points = num_points

        # -- ME -- iterate on list of sequences
        for i in range(len(dataset_list)):
            # -- ME --
            # `num_sequences_in_dataset` represents the number of windows in a sequence
            num_sequences_in_dataset = len(dataset_list[i]) - self.sequence_len * self.stride + 1
            # -- ME -- `dataset_sizes` cumulates the number of windows 
            self.dataset_sizes.append(self.dataset_sizes[i] + num_sequences_in_dataset)
        # -- ME -- `size` is the total number of windows in the whole dataset
        self.size = self.dataset_sizes[-1]
        self.sequence_ids = sequence_ids


    def find_dataset_with_idx(self, idx):
        assert idx < self.size, "INVALID ID"
        # -- ME --
        # returns all the sequences such as the cumulative number of windows is bellow `idx`
        dataset_idx, sizes = list(x for x in enumerate(self.dataset_sizes) if x[1] <= idx)[-1]
        return self.datasets[dataset_idx], idx - sizes, self.sequence_ids[dataset_idx]


    def load_sequence(self, dataset, indices):
        """
        Inputs:
            * `dataset` is a torch.utils.data.Dataset.
            It can be a `KITTI360Sequence`.
            * `indices`: the indices of the frames to load.
        Returns:
            * list of frames: list of data dicts
        """
        sequence = []
        for seq_index in indices:
            data_dict = dataset[seq_index]
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            # -- ME --
            if self.num_points > 0:
                data_dict = self.crop(data_dict)
            sequence.append(data_dict)
        return sequence
    

    def transform_sequence(self, elem):
        if self.sequence_transforms:
            elem = self.sequence_transforms(elem)
        return elem
    
    # -- ME --
    def crop(self, elem):
        
        if isinstance(elem, dict):
            data_dict_croped = {}
            for i, key in enumerate(elem.keys()):
                if "numpy" in key:
                    indices = np.arange(len(elem[key]))
                    if i == 0:
                        if len(elem[key]) >= self.num_points:
                            sample_idx = np.random.choice(indices, self.num_points, replace=False)
                        else:
                            sample_idx = np.concatenate((indices, np.random.choice(indices, self.num_points - len(indices), replace=True)), axis=-1)
                    if len(elem[key].shape) > 1:
                        data_dict_croped[key] = elem[key][sample_idx,:]
                    else:
                        data_dict_croped[key] = elem[key][sample_idx]
                else:
                    data_dict_croped[key] = elem[key]
        else:
            warnings.warn('sequence_dataset.py: crop function -> elem is not dict')
            data_dict_croped = elem
        return data_dict_croped
    

    def __getitem__(self, idx):
        """
        Input: `idx` is the window index
        Return: window -- data_dict containing window's frames data
        """

        dataset, start_idx_in_dataset, seq_id = self.find_dataset_with_idx(idx)
        indices = [start_idx_in_dataset + i * self.stride for i in range(self.sequence_len)]

        # `sequence` list of data_dict
        sequence = self.load_sequence(dataset, indices)
        # `sequence_item` one data_dict constructed from the list of data_dicts
        sequence_item = self.__sequence_collate(sequence)

        sequence_item = self.transform_sequence(sequence_item)
        return sequence_item
    

    def __len__(self):
        return self.size
    

    @staticmethod
    def __sequence_collate(batch):
        """
        Agglomerate window data for a sequence

        Args:
            batch (List): A list of elements which are to be aggregated into a batch of elements
        """
        elem = batch[0]
        if elem is None:
            return batch
        if isinstance(elem, collections.Mapping):
            result = dict()
            for key in elem:
                if "numpy" in key:
                    for idx, d in enumerate(batch):
                        result[f"{key}_{idx}"] = d[key]
                else:
                    # -- ME -- for example key='vertex_map'
                    result[key] = DatasetOfSequences.__sequence_collate([d[key] for d in batch])
            return result
        elif isinstance(elem, np.ndarray):
            # -- ME -- for example key='vertex_map'
            # result['vertex_map']: (batch_size, C, height, width) e.g batch_size=seq_length
            return np.concatenate([np.expand_dims(e, axis=0) for e in batch], axis=0)
        else:
            return default_collate(batch)


    @staticmethod
    def _params_type() -> str:
        return DatasetOfSequences.__name__
