import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from torch._C import device
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from tqdm import tqdm
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
from alficore.dataloader.fgvc_loader import FGVC_dataloader
import matplotlib.pyplot as plt
import time
import os
import copy
from itertools import product
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from typing import Dict, List


cudnn.benchmark = True
plt.ion()   # interactive mode
# logging.config.fileConfig('fi.conf')
# log = logging.getLogger()
cuda_device = 1
model_name = 'resnet'
transform = transforms.Compose([            #[1]
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
            std=[0.229, 0.224, 0.225]                  #[7]
            )])


class build_objdet_native_model(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device):
        super().__init__(model=model)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.postprocess = False
        self.preprocess = True
        self.device = device
        
        # auto_load_resume(model,  "/home/fgeissle/ranger_repo/ranger/image_classification/MMAL/air_epoch146.pth", status='test') #load actual pretrained parameters!
        self.model = model

    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """


        # if 'coco' in self.dataset_name or self.dataset_name == 'robo':
        #     images = [resize(x['image'], self.img_size) for x in batched_inputs] # coco
        #     images = [x/255. for x in images]
        # elif 'kitti' in self.dataset_name:
        #     images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]  # kitti
        #     images = [x/255. for x in images]
        # elif 'lyft' in self.dataset_name:
        #     images = [x["image"]/255. for x in batched_inputs]
        #     # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        #     # Pad to square resolution
        #     padded_imgs = [pad_to_square(img, 0)[0] for img in images]
        #     # Resize
        #     images = [resize(img, self.img_size) for img in padded_imgs]
        # else:
        #     print('dataset_name not known, aborting.')
        #     sys.exit()
  
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        input_tensor = preprocess(batched_inputs)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        

        # # images = [letterbox(x["image"], self.img_size, HWC=True, use_torch=True)[0] for x in batched_inputs]
        # images = [resize(x['image'], self.img_size) for x in batched_inputs]
        # images = [x/255. for x in images]

        # Convert to tensor
        images = torch.stack(images).to(self.device)
        ## normalisde the input if neccesary
        return images

    def postprocess_output(self, output):
        """
        the returning output should be stored as dictionary
        Output['instances] = fields containing pred_boxes, scores, classes
        viz. it should align to attributes as used in function instances_to_coco_json() in coco evaluation file.
        Output['instances].pred_boxes = [[2d-bb_0], [2d-bb_1], [2d-bb_2]...]
        Output['instances].scores     = [score_0, score_1, .....]
        Output['instances].classes     = [car, pedetrian, .....]

        ex: for batch size 1
        Output = [{}]
        Output['instances'] = output
        return Output        
        """
        output = output[-2]
        # output = output[-2].max(1, keepdim=True)[1].T

        return output

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:          
            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            def wrapper(*args, **kwargs):            
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            return wrapper
        except KeyError:
            raise AttributeError(method)


    def __call__(self, input, dummy=False):
        # input = pytorchFI_objDet_inputcheck(input)

        self.model = self.model.to(self.device)
        input = input.to(self.device)

        _input = input
        if self.preprocess:
            _input = self.preprocess_input(input)

        output = self.model(_input, DEVICE=self.device)

        if self.postprocess:
            output = self.postprocess_output(output)

        # output = pytorchFI_objDet_outputcheck(output)
        return output


device = torch.device(
        "cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")





def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.detach().cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(model, images, labels, class_names):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(model, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(24, 24))
    for idx in np.arange(3):
        ax = fig.add_subplot(1, 3, idx+1, xticks=[], yticks=[])
        try:
            matplotlib_imshow(images[idx], one_channel=False)
        except:
            x=0
        try:
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                class_names[preds[idx]],
                probs[idx] * 100.0,
                class_names[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"), fontsize=36)
        except:
            x = 0
    return fig

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_num_correct(preds, labels):
    return preds.argmax(dim=0).eq(labels).sum().item()

class DNN_TRAIN_HYPER_PARAMETERS:
    """
    DNN_TRAIN_HYPER_PARAMETERS: hyper_parameters
    """
    def __init__(self, lr=0.001, momentum=0.9, gamma=0.1, batch_size = 32, shuffle=False, step_size=10, path=None, epochs=1, checkpoint=5) -> None:
        self.lr         = lr
        self.momentum   = momentum
        self.gamma      = gamma
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.step_size  = step_size
        self.path       = path
        self.epochs     = epochs
        self.checkpoint = checkpoint

def train_model(model, criterion, optimizer, scheduler, mini_batch_num=1, tb_writer=None, dnn_hyper_parameters=None, dataloaders=None, dataset_sizes=None, class_names=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(dnn_hyper_parameters.epochs)):
        print(f'Epoch {epoch}/{dnn_hyper_parameters.epochs - 1}')
        print('-' * 10)
        epoch_loss = 0
        epoch_acc = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            dataloaders[phase].datagen_reset()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            mini_batch = {'train': 0, 'val':0}

            with tqdm(total=dataloaders[phase].dataset_length) as pbar:
                while dataloaders[phase].data_incoming:
                    pbar.set_description(phase)
                # for inputs, labels in dataloaders[phase]:
                    dataloaders[phase].datagen_itr()
                    inputs, labels = dataloaders[phase].images, dataloaders[phase].labels
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    batch_acc = get_num_correct(preds, labels)
                    epoch_acc += batch_acc
                    
                    # if mini_batch[phase] % mini_batch_num*dataloaders[phase].curr_batch_size == mini_batch_num*dataloaders[phase].curr_batch_size - 1:
                    """
                    Use this to store the check points for every mini_batch_num
                    """
                    mini_batch[phase] = mini_batch[phase] + 1
                    pbar.update(dataloaders[phase].curr_batch_size)
                    if tb_writer is not None:
                        # ...log a Matplotlib Figure showing the model's predictions on a
                        # random mini-batch
                        tb_writer.add_scalar('{} batch loss'.format(phase), batch_loss/dataloaders[phase].curr_batch_size, epoch*int(dataset_sizes[phase]/dataloaders[phase].batch_size) + mini_batch[phase])
                        # tb_writer.add_scalar('{} correct'.format(phase), epoch_acc, epoch*sint(dataset_sizes[phase]/dataloaders[phase].batch_size) + mini_batch[phase])
                        tb_writer.add_scalar('{} batch Accuracy'.format(phase), batch_acc/dataloaders[phase].curr_batch_size, epoch*int(dataset_sizes[phase]/dataloaders[phase].batch_size) + mini_batch[phase])
                dataloaders[phase].datagen_reset()
            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss / dataset_sizes[phase]
            epoch_acc = epoch_acc/ dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            tb_writer.add_scalar('{} epoch loss'.format(phase), epoch_loss/dataset_sizes[phase], epoch)
            # tb_writer.add_scalar('{} correct'.format(phase), epoch_acc, epoch*sint(dataset_sizes[phase]/dataloaders[phase].batch_size) + mini_batch[phase])
            tb_writer.add_scalar('{} epoch Accuracy'.format(phase), epoch_acc/dataset_sizes[phase], epoch)
            if phase == 'val':
                tb_writer.add_figure('{} epoch - predictions vs. actuals'.format(phase), plot_classes_preds(model, inputs, labels, class_names=class_names), global_step=epoch)
            dataloaders[phase].datagen_reset()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train' and epoch%dnn_hyper_parameters.checkpoint == 0:
                save_path = os.path.join(dnn_hyper_parameters.path, "checkpoints", 'model_weights_{}.pth'.format(epoch))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, save_path)
                print("saved checkpoint at epoch {} in {}".format(epoch, save_path))

        print('Finished epoch {}'.format(epoch))
    tb_writer.add_hparams(
            {"lr": dnn_hyper_parameters.lr, "batch_size": dnn_hyper_parameters.batch_size, "shuffle":dnn_hyper_parameters.shuffle, "gamma":dnn_hyper_parameters.gamma, \
                "momentum":dnn_hyper_parameters.momentum, "step_size":dnn_hyper_parameters.step_size},
            {
                "accuracy": epoch_acc,
                "loss": epoch_loss,
            },
        )

    tb_writer.close()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(dnn_hyper_parameters.path, 'model_weights_final.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("saved final model in {}".format(save_path))

def do_fine_tuning(epochs=25, mini_batch_num=1, tb_writer=None, dnn_hyper_parameters:DNN_TRAIN_HYPER_PARAMETERS=None):
    """
        ### Finetuning the convnet ###
    """
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = dnn_hyper_parameters.batch_size
    dl_attr.dl_shuffle     = dnn_hyper_parameters.shuffle
    dl_attr.dl_sampleN     = 1000
    dl_attr.dl_num_workers = 1
    dl_attr.dl_device      = device
    dl_attr.dl_dataset_name  = "fgcv"
    dl_attr.dl_dataset_type  = "val" 
    dl_attr.dl_transform     = transform

    val_fgcv_dataloader = FGVC_dataloader(dl_attr=dl_attr)

    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = dnn_hyper_parameters.batch_size
    dl_attr.dl_shuffle     = dnn_hyper_parameters.shuffle
    dl_attr.dl_sampleN     = 1.0
    dl_attr.dl_num_workers = 4
    dl_attr.dl_device      = device
    dl_attr.dl_dataset_name  = "fgcv"
    dl_attr.dl_dataset_type  = "train" 
    dl_attr.dl_transform     = transform

    train_fgcv_dataloader = FGVC_dataloader(dl_attr=dl_attr)

    dataloaders = {'train': train_fgcv_dataloader, 'val':val_fgcv_dataloader}
    dataset_sizes = {'train': train_fgcv_dataloader.dataset_length, 'val':val_fgcv_dataloader.dataset_length}
    class_names = dataloaders['train'].classes

    model_ft = models.resnet50(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=dnn_hyper_parameters.lr, momentum=dnn_hyper_parameters.momentum, nesterov=True)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=dnn_hyper_parameters.step_size, gamma=dnn_hyper_parameters.gamma)

    """
        ### Train and evaluate ###
    """

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs, mini_batch_num=mini_batch_num, tb_writer=tb_writer, dnn_hyper_parameters=dnn_hyper_parameters, dataloaders=dataloaders, \
        dataset_sizes=dataset_sizes, class_names=class_names)


def do_transfer_learning(mini_batch_num=1, tb_writer=None, dnn_hyper_parameters:DNN_TRAIN_HYPER_PARAMETERS=None):
    """
        ### Transfer learning ###
        ConvNet as fixed feature extractor
    """
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = dnn_hyper_parameters.batch_size
    dl_attr.dl_shuffle     = dnn_hyper_parameters.shuffle
    dl_attr.dl_sampleN     = 1000
    dl_attr.dl_num_workers = 1
    dl_attr.dl_device      = device
    dl_attr.dl_dataset_name  = "fgcv"
    dl_attr.dl_dataset_type  = "val" 
    dl_attr.dl_transform     = transform

    val_fgcv_dataloader = FGVC_dataloader(dl_attr=dl_attr)

    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = False
    dl_attr.dl_batch_size  = dnn_hyper_parameters.batch_size
    dl_attr.dl_shuffle     = dnn_hyper_parameters.shuffle
    dl_attr.dl_sampleN     = 1.0
    dl_attr.dl_num_workers = 4
    dl_attr.dl_device      = device
    dl_attr.dl_dataset_name  = "fgcv"
    dl_attr.dl_dataset_type  = "train" 
    dl_attr.dl_transform     = transform

    train_fgcv_dataloader = FGVC_dataloader(dl_attr=dl_attr)

    dataloaders = {'train': train_fgcv_dataloader, 'val':val_fgcv_dataloader}
    dataset_sizes = {'train': train_fgcv_dataloader.dataset_length, 'val':val_fgcv_dataloader.dataset_length}
    class_names = dataloaders['train'].classes

    model_conv  = models.resnet50(pretrained=True)
    model_conv.load_state_dict(torch.load('/home/fgeissle/ranger_repo/ranger/image_classification/MMAL/resnet50-19c8e357.pth'))

    num_ftrs = model_conv.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv = model_conv.to(device)
    # model_conv = build_objdet_native_model(model_conv, device) #added input processing here


    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=dnn_hyper_parameters.lr, momentum=dnn_hyper_parameters.momentum, nesterov=True)
    # optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=dnn_hyper_parameters.lr, momentum=dnn_hyper_parameters.momentum, nesterov=True)
    # optim.Adamax, optim.Adam

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=dnn_hyper_parameters.step_size, gamma=dnn_hyper_parameters.gamma)

    """
        ### Train and evaluate ###
    """

    train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, mini_batch_num=mini_batch_num, tb_writer=tb_writer, dnn_hyper_parameters=dnn_hyper_parameters, dataloaders=dataloaders, \
        dataset_sizes=dataset_sizes, class_names=class_names)


# parameters = dict(
#     lr = [0.01, 0.099, 0.098],
#     momentum = [0.9, 0.9, 0.8],
#     gamma = [0.1, 0.099, 0.098],
#     step_size = [10, 15, 20],
#     batch_size = [64,8,12],
#     shuffle = [True, True, True]
# )
parameters = dict(
    lr = [1.],
    momentum = [0.9],
    gamma = [0.1],
    step_size = [10],
    batch_size = [8],
    shuffle = [True]
)


param_values = [v for v in parameters.values()]
experiment = time.strftime("%Y%m%d-%H%M%S")
for run_id, (lr, momentum, gamma, step_size, batch_size, shuffle) in enumerate(product(*param_values)):
    print("run id:", run_id + 1)
    epochs = 20
    experiment_path = 'train/fgvc_experiment_{}/{}'.format(experiment, run_id)
    dnn_hyper_parameters = DNN_TRAIN_HYPER_PARAMETERS(lr=lr, momentum=momentum, gamma=gamma, batch_size = batch_size, step_size=step_size, shuffle=shuffle,\
        epochs=epochs, path=experiment_path, checkpoint=2)
    experiment = run_id
    mini_batch_num = 5
    
    os.makedirs(experiment_path, exist_ok=True)
    tb_writer = SummaryWriter(experiment_path)
    do_transfer_learning(mini_batch_num=mini_batch_num, tb_writer=tb_writer, dnn_hyper_parameters=dnn_hyper_parameters)
    # do_fine_tuning(epochs=2)




# plt.ioff()
# plt.show()
