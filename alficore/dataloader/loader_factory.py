from .cifar10_loader import Cifar10_Loader


class LoaderFactory:
    @staticmethod
    def get_dataloader(loader_name):
        if loader_name.lower() == "cifar10":
            return Cifar10_Loader()
        # add more data loader classes in directory dataloader and
        # add instantiation below
        else:
            raise NotImplementedError