from torchvision import datasets, transforms
import torch
import numpy as np
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr

class MNIST_dataloader():
    def __init__(self, dl_attr:TEM_Dataloader_attr):
    # def __init__(self, **kwargs):  
        """
        Args:
        dataset_type (str) : Type of dataset shall be used - train, test, val... Defaults to val.
        batch_size   (uint): batch size. Defaults to 1.
        shuffle      (str) : Shuffling of the data in the dataloader. Defaults to False.
        sampleN      (uint): Percentage of dataset lenth to be sampled. Defaults to None.
        transform  (obj) : Several transforms composed together. Defaults to predefined transform.
        num_workers  (uint): Number of workers to be used to load the dataset. Defaults to 1.
        device       (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPuU.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """        
        self.dataset_type = dl_attr.dl_dataset_type
        self.batch_size = dl_attr.dl_batch_size
        self.shuffle = dl_attr.dl_shuffle
        self.sampleN = dl_attr.dl_sampleN
        # self.transform = dl_attr.dl_transform
        self.transform = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x*255)
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                ])
        self.num_workers = dl_attr.dl_num_workers
        self.device = dl_attr.dl_device

        if self.dataset_type == 'train':
        # (self.loader, self.test_loader, self.classes) = load_data_mnist(batch_size, test_batch_size)
            self.dataset = datasets.MNIST('LeNet5/data', train=True, download=True,
                                transform=self.transform)
            # self.data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        elif self.dataset_type == 'test' or self.dataset_type == 'val':
        # Fetch test data: total 10000 samples
            self.dataset = datasets.MNIST('LeNet5/data', train=False, download=True, transform=self.transform)
        if self.sampleN is not None:
            if self.sampleN < 1:
                self.sampleN = int(self.sampleN*len(self.dataset))
                print('Set dataset len to', self.sampleN)

            self.dataset = torch.utils.data.Subset(self.dataset, np.random.choice(len(self.dataset), self.sampleN, replace=False)) #random n elements
            # self.dataset = torch.utils.data.Subset(self.dataset, np.array(np.linspace(3,len(self.dataset),self.sampleN)).astype(int)) #specific n elements starting from 3
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.datasetfp = None
        self._image_path = -1
        
        self.data_incoming = True
        self.images = -1
        self.labels = -1    

        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.dataset)
        self.datagen_iter_cnt = 0
        print("MNIST Dataset loaded in {}; {} dataset - Length : {}".format(self.device, self.dataset_type, len(self.dataset)))

    def datagen_itr(self):
        if self.datagen_iter_cnt < self.dataset_length:
            self.data_incoming = True
            self.images, self.labels = self.datagen_iter.next()
            self.image_path = [self._image_path for i in range(len(self.images))]
            self.curr_batch_size = len(self.images)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            self.images = self.images.to(self.device)
            self.labels = self.labels.to(self.device)
            if self.datagen_iter_cnt == self.dataset_length: # added to stop repeat of last batch of images
                self.data_incoming = False
        else:
            self.data_incoming = False

    def datagen_reset(self):
        self.images = -1
        self.labels = -1
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0
