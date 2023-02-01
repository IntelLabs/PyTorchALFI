"""
Abstract class to define necessary functions for each data loader

"""
from abc import ABC, abstractmethod

class Abstract_Loader(ABC):
    @abstractmethod
    def get_test_loader(self, data_dir,
                        batch_size,
                        shuffle,
                        valid_size,
                        num_workers,
                        pin_memory,
                        std,
                        mean):
        """
        get loader for test data
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_size(self):
        raise NotImplementedError

