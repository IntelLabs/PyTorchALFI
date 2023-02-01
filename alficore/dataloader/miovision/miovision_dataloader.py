import torch
import pandas as pd
import os, pickle
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


get_label_for_training = {
    'articulated_truck': 0,
    'background': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'motorcycle': 5,
    'non-motorized_vehicle': 6,
    'pedestrian': 7,
    'pickup_truck': 8,
    'single_unit_truck': 9,
    'work_van': 10
}
get_label_for_testing = dict((v, k) for k, v in get_label_for_training.items())

class_clustering_dict = {'vulnerable': ['pedestrian', 'bicycle'], 'non_vulnerable': ['articulated_truck', 'bus', 'car', 
                        'motorcycle', 'non-motorized_vehicle', 'pickup_truck', 'single_unit_truck', 'work_van'],
                        'background': ['background']}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': sample['label']}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': sample['label']}


class DS(Dataset):
    '''
    This class reads image with its provided name and location from mapping file, the location of the dataset folder
    must be arranged in respective class sub-folders, returns a dictionary of the image and its label
    '''
    def __init__(self, csv_file, transform_func, parser):
        self.transform = transform_func
        self.csv_ds = pd.read_csv(csv_file, dtype='str', header=None)
        self.parser = parser

    def __len__(self):
        return len(self.csv_ds)

    def __getitem__(self, idx):
        # to prepare train val set with labels
        if not self.parser.evaluate:
            if not self.parser.val_correct:
                dir_name = 'train'
            else:
                dir_name = 'correct_val_set_' + 'resnet50' # self.parser.model_name 
            img_path = os.path.join(self.parser.root_directory, dir_name, str(self.csv_ds.iloc[idx][1]),
                                    str(self.csv_ds.iloc[idx][0]) + '.jpg')
            class_name = self.csv_ds.iloc[idx][1]                        
            label = get_label_for_training[class_name]
            
        # to prepare test set without labels
        else:
            img_path = os.path.join(self.root_dir, 'test', str(self.csv_ds.iloc[idx][0]) + '.jpg')
            label = ''
        image = io.imread(img_path)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        sample['image_path'] =  img_path
        return sample


def generate_train_indices_ranger(total_indices, data_file, train_indices_checkpoint, sampleN_percent=20):
    """Prepares indices of training data that would be used for generating ranger bounds

    Args:
        total_indices (List): Train set indices list
        data_file (str): CSV file path containing train set mapping between image and label
        train_indices_checkpoint (str): Pickle file path of generated indices checkpoint
        sampleN_percent (int, optional): Percentage of training data to be used. Defaults to 20.

    Returns:
        [type]: [description]
    """
    index_labels = {key:[] for key in get_label_for_training.keys()}
    ranger_indices_classes = {key:[] for key in get_label_for_training.keys()}
    ranger_indices_classes['overall'] = []
    # file containing a dictionary with class names as keys and list of training indices as values
    saved_file_path = os.getcwd() + train_indices_checkpoint
    csv_df = pd.read_csv(data_file, dtype='str', header=None)

    if os.path.isfile(saved_file_path):
        with open(saved_file_path, 'rb') as f:
            index_labels = pickle.load(f)
    else:
        for idx in sorted(total_indices):
            label = str(csv_df.iloc[idx][1])
            label_id_list = index_labels[label]
            label_id_list.append(idx)
        with open(saved_file_path, 'wb') as f:
            pickle.dump(index_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    for class_name in index_labels.keys():
        class_ids = index_labels[class_name]
        ranger_indices_classes[class_name].extend(np.random.choice(class_ids, int(sampleN_percent*len(class_ids)/100), replace=False))
    
    ranger_indices_classes['overall'].extend(np.random.choice(total_indices, int(sampleN_percent*len(total_indices)/100), replace=False))
    return ranger_indices_classes
    

def generate_fault_injection_indices(val_indices, data_file, parser):
    """Prepares indices of validation data (uniformly from each class) that would be used for fault injection

    Args:
        val_indices (List): Validation set indices list
        data_file (str): CSV file path containing correctly classified validation set mapping between image and label
        parser (ConfigParser): Config of miovision.yml

    Returns:
        [List]: [Indices of validation set to be used for fault injection]
    """
    index_labels = {key:[] for key in get_label_for_training.keys()}
    fault_injection_indices = []
    saved_file_path = ''
    if parser.fault_type == 'weights_injs':
        # file containing val indices for pytorchfi fault injection
        saved_file_path = os.getcwd() + parser.val_indices_fault_inj_weights_checkpoint
    if os.path.isfile(saved_file_path):
        with open(saved_file_path, 'rb') as f:
            fault_injection_indices = pickle.load(f)
    else:
        csv_df = pd.read_csv(data_file, dtype='str', header=None)
        for idx in sorted(val_indices):
            label = str(csv_df.iloc[idx][1])
            label_id_list = index_labels[label]
            label_id_list.append(idx)

        if parser.fault_type == 'weights_injs':
            for class_name in index_labels.keys():
                class_ids = index_labels[class_name]
                fault_injection_indices.extend(np.random.choice(class_ids, parser.number_of_images_each_class, replace=False))

            with open(saved_file_path, 'wb') as f:
                pickle.dump(fault_injection_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif parser.fault_type == 'neurons_injs':
            class_ids = np.random.choice(index_labels[get_label_for_testing[parser.neuron_inj_class_id]], parser.number_of_images_each_class, replace=False)
            fault_injection_indices.extend(class_ids)

    return fault_injection_indices


def generate_uniform_class_indices(parser):
    train_indices = []

    # loading the checkpoint which stores the dictionary of each class to number of train images mapping
    saved_file_path = os.getcwd() + parser.train_class_indices_checkpoint
    with open(saved_file_path, 'rb') as f:
        index_labels = pickle.load(f)
    
    # fetching the minimum number of images among all the miovision classes
    min_images_class = len(index_labels[sorted(index_labels, key=lambda key: len(index_labels[key]))[0]])
    for class_name in index_labels.keys():
        class_ids = index_labels[class_name]
        train_indices.extend(np.random.choice(class_ids, min_images_class, replace=False))

    return train_indices


def miovision_data_loader(parser, batch_size=500):
    """Prepares the miovision torch data loaders for train, val and test set

    Args:
        parser (ConfigParser): Config of miovision.yml
        batch_size (int, optional): Batch size of dataloader. Defaults to 500.

    Returns:
        [Torch dataloader]: train, validation, test dataloaders
    """
    train_loader = None
    validation_loader = None
    test_loader = None
    train_loaders = dict()
    np.random.seed(parser.random_seed)
    if not parser.evaluate:
        if not parser.val_correct or parser.train:
            tds = DS(parser.train_ground_truth_file, 
                    transforms.Compose([Rescale((parser.rescale_dimension, parser.rescale_dimension)), ToTensor()]),
                    parser)
            dataset_size = len(tds)
            indices = list(range(dataset_size))
            val_slice = int(np.floor(parser.validation_split * dataset_size))
            if parser.shuffle_dataset:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[val_slice:], indices[:val_slice]
            # evaluates dataloader for generating ranger bounds
            if parser.generate_ranger_bounds:
                train_indices_dict = generate_train_indices_ranger(train_indices, parser.train_ground_truth_file, 
                                                                parser.train_class_indices_checkpoint)
                # evaluates dataloader for generating ranger bounds picking 20% from each class training data
                if not parser.generate_ranger_bounds_classes:
                    train_indices = train_indices_dict['overall']
                    train_sampler = SubsetRandomSampler(train_indices)
                    train_loader = torch.utils.data.DataLoader(tds, batch_size=parser.batch_size, sampler=train_sampler)                    
                # evaluates dataloader for generating ranger bounds picking 20% of entire training set
                else:
                    for class_name in train_indices_dict.keys():
                        train_indices = train_indices_dict[class_name]
                        train_sampler = SubsetRandomSampler(train_indices)
                        train_loader = torch.utils.data.DataLoader(tds, batch_size=parser.batch_size, sampler=train_sampler)                        
                        train_loaders[class_name] = train_loader 
                    train_loader = train_loaders
            elif parser.train_uniform_class_dist:
                train_indices = generate_uniform_class_indices(parser)
                train_sampler = SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(tds, batch_size=parser.batch_size, sampler=train_sampler)       
            # evaluates dataloader (train_loader) for training i.e. using entire training set and dataloader (validation_loader) for entire validation set
            else: 
                train_sampler = SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(tds, batch_size=parser.batch_size, sampler=train_sampler)                
            validation_sampler = SubsetRandomSampler(val_indices)
            validation_loader = torch.utils.data.DataLoader(tds, batch_size=parser.batch_size, sampler=validation_sampler)
        # evaluates validation dataloader using correctly classified images by the original model
        elif parser.val_correct:
            correct_val_tds = DS(parser.correct_val_ground_truth_file, 
                transforms.Compose([Rescale((parser.rescale_dimension, parser.rescale_dimension)), ToTensor()]),
                parser)
            ds_size = len(correct_val_tds)
            indices = list(range(ds_size))
            # evaluates dataloader for fault injection experiments 
            if parser.inject_fault:
                indices = generate_fault_injection_indices(indices, parser.correct_val_ground_truth_file, 
                                        parser)
            correct_validation_sampler = SubsetRandomSampler(indices)
            validation_loader = torch.utils.data.DataLoader(correct_val_tds, batch_size=batch_size, sampler=correct_validation_sampler, shuffle=False)
    # evaluates test dataloader to evaluate on test set without having labels
    else:
        test_ds = DS(parser.test_image_names, 
                 transforms.Compose([Rescale((parser.rescale_dimension, parser.rescale_dimension)), ToTensor()]),
                parser)
        dataset_size = len(test_ds)
        indices = list(range(dataset_size))
        test_indices = np.random.choice(indices, parser.num_test_images)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=parser.num_test_images, sampler=test_sampler)

    return train_loader, validation_loader, test_loader


def class_clustering(class_name):
    """Gives the cluster name in which a class belongs

    Args:
        class_name (str): Class name of miovision image classification dataset

    Returns:
        [str]: Cluster name
    """
    
    for cluster in class_clustering_dict.keys():
        if class_name in class_clustering_dict[cluster]:
            cluster_name = cluster

    return cluster_name


def safety_critical_confusion(ground_truth_class, predicted_misclassified_class):
    """Logic to decide safety-criticality of SDC

    Args:
        ground_truth_class (List): Ground truth labels
        predicted_misclassified_class (List): Corrupted model prediction labels
    Returns:
        [List]: List of flags indicating safety-criticality, True if safety-critical, False if not
    """
    safety_critical_bool = []

    for i in range(len(ground_truth_class)):
        safety_critical_sdc = False
        ground_truth_class_cluster = class_clustering(get_label_for_testing[ground_truth_class[i]])
        predicted_misclassified_class_cluster = class_clustering(get_label_for_testing[predicted_misclassified_class[i][0]])

        if (ground_truth_class_cluster == 'vulnerable' and predicted_misclassified_class_cluster != 'vulnerable') \
                                or (ground_truth_class_cluster == 'non_vulnerable' and predicted_misclassified_class_cluster == 'background'):
            safety_critical_sdc = True

        safety_critical_bool.append(safety_critical_sdc)

    return safety_critical_bool


class Miovision_dataloader():

    def __init__(self, **kwargs):  
        """
        Args:
        dataset_parser (ConfigParser) : Miovision configuration parser
        device       (uint): Cuda/Cpu device to be used to load dataset. Defaults to Cuda 0 if available else CPU.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """        
        self.parser = kwargs.get("dataset_parser", None)
        self.batch_size = kwargs.get("batch_size", 50)
        self.device = kwargs.get("device", torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu"))
        _, val_loader, _ = miovision_data_loader(self.parser, self.batch_size)
        self.data_loader = val_loader
        self.data_incoming = True
        self.datasetfp = []
        for _, x in enumerate(self.data_loader):
            self.datasetfp.extend(x['image_path'])
        self.datagen_iter = iter(self.data_loader)
        self.dataset_length = len(self.data_loader.sampler)
        self.datagen_iter_cnt = 0
        print("Miovision Dataset loaded in {}; {} dataset - Length : {}".format(self.device, 'val', self.dataset_length))


    def datagen_itr(self):
        self.data_incoming = True
        if self.datagen_iter_cnt < self.dataset_length:
            data = self.datagen_iter.next()
            self.images = Variable(data['image']).float().to(self.device)
            self.labels = Variable(data['label']).to(self.device)
            self.image_path = data['image_path']
            self.curr_batch_size = len(self.images)
            self.datagen_iter_cnt = self.datagen_iter_cnt + self.curr_batch_size
            if self.datagen_iter_cnt == self.dataset_length: # added to stop repeat of last batch of images
                self.data_incoming = False
        else:
            self.data_incoming = False


    def datagen_reset(self):
        self.datagen_iter = iter(self.data_loader)
        self.data_incoming = True
        self.datagen_iter_cnt = 0

        

