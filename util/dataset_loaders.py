import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.utils.data as data
import json


def load_data_mnist(train_batch_size, test_batch_size):
    # Fetch training data: total 60000 samples
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False)

    # Fetch test data: total 10000 samples
    testset = datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print('MNIST Dataset loaded. Length', 'train', len(trainset), 'test', len(testset))
    return (train_loader, test_loader, classes)



def load_data_cifar(train_batch_size, test_batch_size):

    # Batch normalization (rescale the values to a common range)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalize to tensors of range [-1,1]

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) #For training
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=0)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) #For testing
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0) #random: shuffle=True, same:shuffle=False

    print('CIFAR10 Dataset loaded. Length', 'train', len(trainset), 'test', len(testset))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (train_loader, test_loader, classes)



def load_data_imageNet_piece(test_batch_size):
    """
    Loads a piece of imageNet if arranged appropriately in classes subfolders.
    :param test_batch_size:
    :return: test_loader,
    :return: classes_orig is the original list of 1000 classes in same order as json file (used for normal predictions)
    :return: classes_sorted is list of reduced (here 641) classes in the order the DataLoader uses (used for GT)
    """
    # below works for vgg16, alexnet, resnet50(?)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    imagenet_piece = datasets.ImageFolder(root='./data/imagenet_val',
                                          transform=data_transform)
    imagenet_loader = data.DataLoader(imagenet_piece, batch_size=test_batch_size, shuffle=True, num_workers = 0)

    class_idx = json.load(open("imagenet_class_index.json"))
    classes_orig = [class_idx[str(k)][1] for k in range(len(class_idx))]

    classes_sorted = list(imagenet_piece.class_to_idx.keys())

    return (imagenet_loader, classes_orig, classes_sorted)





# Code for loading a subset: --------------------------------------

def load_Subtraindata_mnist(train_batch_size, p):
    """
    :param test_batch_size: batchsize for dataloader
    :param p: portion of testset, 0...1
    :return: subset testloader
    """
    # Fetch train data: total 50000 samples
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))


    portion = list(range(0, int(trainset.__len__()*p)))
    trainset2 = torch.utils.data.Subset(trainset, portion)

    train_loader = torch.utils.data.DataLoader(trainset2, batch_size=train_batch_size, shuffle=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print('MNIST Data-Subset loaded. Length', 'test', len(trainset2))
    return (train_loader, classes)



def load_Subtestdata_imageNet(test_batch_size, path, nn):
    """
    Loads a piece of imageNet if arranged appropriately in classes subfolders.
    :param test_batch_size:
    :return: test_loader,
    :return: classes_orig is the original list of 1000 classes in same order as json file (used for normal predictions)
    :return: classes_sorted is list of reduced (here 641) classes in the order the DataLoader uses (used for GT)
    """

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    testset = datasets.ImageFolder(root=path, transform=data_transform)

    if testset.__len__() >= nn:
        portion = list(range(0, nn))
    else:
        portion = list(range(0, testset.__len__()))

    testset2 = torch.utils.data.Subset(testset, portion)
    imagenet_loader = data.DataLoader(testset2, batch_size=test_batch_size, shuffle=False, num_workers = 0)

    class_idx = json.load(open("imagenet_class_index.json"))
    classes_orig = [class_idx[str(k)][1] for k in range(len(class_idx))]
    classes_sorted = list(testset.class_to_idx.keys())

    return (imagenet_loader, classes_orig, classes_sorted)
