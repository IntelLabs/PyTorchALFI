from __future__ import print_function, division

import os, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
import alficore.dataloader.miovision.miovision_config_parser as parser
from alficore.dataloader.miovision.miovision_dataloader import miovision_data_loader as data_loader
from alficore.dataloader.miovision.miovision_dataloader import get_label_for_testing


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience_epochs=7, verbose=False, delta=0, path='resnet50_mio/resnet50_mio_checkpoint.pt',
                 trace_func=print):
        """
        Args:
            patience_epochs (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, validation_loss, trained_model):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(validation_loss, trained_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(validation_loss, trained_model)
            self.counter = 0

    def save_checkpoint(self, validation_loss, trained_model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {validation_loss:.6f}). Saving '
                            f'model ...')
        torch.save(trained_model.state_dict(), self.path)
        self.val_loss_min = validation_loss


# INPUTS: output have shape of [batch_size, category_count]
# and target in the shape of [batch_size] * there is only one true class for each sample
# top_k is tuple of classes to be included in the precision
# top_k have to a tuple so if you are giving one number, do not forget the comma
def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    __TOP_RES = 1
    num_of_images = len(output) 
    correctly_classified_images = 0 
    for i in range(num_of_images):
        _output = output[i]
        _output = torch.unsqueeze(_output, 0)
        # percentage = torch.nn.functional.softmax(_output, dim=1)[0] * 100
        _, output_index = torch.sort(_output, descending=True)
        # output_perct = np.round(percentage[output_index[0][:__TOP_RES]].cpu().detach().numpy(), decimals=2)
        output_index = output_index[0][:__TOP_RES].cpu().detach().numpy()
        if output_index[0] == target[i]:
            correctly_classified_images += 1

    accuracy = (correctly_classified_images * 100) / num_of_images
    # we do not need gradient calculation for those
    with torch.no_grad():
        # we will use biggest k, and calculate all precisions from 0 to k
        max_k = max(top_k)
        # top_k gives biggest max_k values
        # output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
        # input=max_k, so we will select max_k number of classes
        # so result will be [batch_size,max_k]
        # top_k returns a tuple (values, indexes) of results
        # we only need indexes(prediction)
        _, prediction = output.topk(k=max_k, dim=max_k, largest=True, sorted=True)
        # then we transpose prediction to be in shape of [max_k, batch_size]
        prediction = prediction.t()
        # we flatten target and then expand target to be like prediction target [batch_size] becomes [1,batch_size]
        # target [1,batch_size] expands to be [max_k, batch_size] by repeating same correct class answer max_k times.
        # when you compare prediction (indexes) with expanded target, you get 'correct' matrix in the shape of  [max_k,
        # batch_size] filled with 1 and 0 for correct and wrong class assignments
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))
        """ correct=([[0, 0, 1,  ..., 0, 0, 0],
            [1, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 1, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8) """
    res = []
    data_size = len(correct.cpu().numpy()[0].tolist())
    # then we look for each k summing 1s in the correct matrix for first k element.
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / data_size))   
    
    return correct.cpu().numpy()[0].tolist(), res[0].item()
    # return accuracy, res[0].item()


# to load the latest trained model checkpoint
def resnet50_load_checkpoint(model, uniform_class_dict=False):
    # load the last checkpoint with the best model
    if not uniform_class_dict:
        model_path = os.getcwd() + '/miovision_results/resnet50_mio/resnet50_mio_checkpoint.pt'
    else:
        model_path = os.getcwd() + '/miovision_results/resnet50_mio/resnet50_mio_uniform_class_checkpoint.pt'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


# to load the latest trained model checkpoint
def vgg16_load_checkpoint(model):
    # load the last checkpoint with the best model
    model_path = os.getcwd() + '/miovision_results/vgg16_mio/vgg16_mio_checkpoint.pt'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def resnet50_net(num_classes):
    # transfer learning to initiliaze the model parameters with those of pre-trained resnet50 with ImageNet
    model = models.resnet50(pretrained=True, progress=True)
    num_features = model.fc.in_features # features coming out of flatten layer of resnet50 and entering the FC layer
    model.fc = nn.Linear(num_features, num_classes) # Replacing the exisitng fc layer with new fc layer having 11 neurons (classes)
    return model


def vgg16_net(num_classes):
    # transfer learning to initiliaze the model parameters with those of pre-trained vgg16 with ImageNet
    model = models.vgg16(pretrained=True, progress=True)
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add customized layer with miovision classes
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    return model


def separate_correctly_classified_images(val_output_flags, image_paths, correct_val_df, parser):
    dir_name = 'correct_val_set_' + parser.model_name
    for i in range(len(val_output_flags)):
        if val_output_flags[i]:
            source_image_path = image_paths[i]
            class_name = source_image_path.split('/')[-2]
            file_name = source_image_path.split('/')[-1].split('.')[0]
            correct_val_df = correct_val_df.append(pd.DataFrame({'File_name': [file_name], 'Class': [class_name]}))
            target_image_path = os.path.join(parser.root_directory, dir_name, class_name)
            if not os.path.exists(target_image_path):
                os.makedirs(target_image_path)
            shutil.copy(source_image_path, target_image_path) 

    return correct_val_df  


def main():
    cuda_device = 0
    optimize_parameters = None  # params to feed to optimizer
    pretrained_freeze_params = False  # True to freeze all pre-trained resnet50 layers and only feed fc layer parameters
    # to the optimizer, False to feed all model parameters to the optimizer

    config_location = 'alficore/dataloader/miovision/miovision_config.yml' 
    scenario = parser.load_scenario(conf_location=config_location)
    config_parser = parser.ConfigParser(**scenario)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cuda_device))    
    else:
        device = torch.device("cpu")

    model = resnet50_net(config_parser.num_classes) # creating a resnet50 model
    # model = vgg16_net(config_parser.num_classes) # creating a resnet50 model
    model = model.to(device)

    # to track the training loss as the model trains in an epoch
    train_losses = []
    # to track the validation loss as the model is evaluated in an epoch
    valid_losses = []
    # to track the validation accuracy as the model is evaluated in an epoch
    validation_accuracies = []
    # to track the test loss
    val_accuracies = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # to perform training or evaluating with validation data (calculating accuracy) having ground truth labels
    if not config_parser.evaluate:    
        train_loader, validation_loader, _ = data_loader(parser=config_parser)    
        if config_parser.train:
            # to optimize only the final fc layer
            if pretrained_freeze_params:
                optimize_parameters = model.fc.parameters()
            # to optimize the entire model
            else:
                optimize_parameters = model.parameters()
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            if config_parser.model_name == 'resnet50':
                optimizer = torch.optim.SGD(params=optimize_parameters, lr=config_parser.learning_rate, momentum=config_parser.momentum,
                                            weight_decay=config_parser.weight_decay)
                # initialize the early_stopping object
                if not config_parser.train_uniform_class_dist:
                    early_stopping = EarlyStopping(patience_epochs=config_parser.patience, verbose=True)
                else:
                    early_stopping = EarlyStopping(patience_epochs=config_parser.patience, verbose=True, 
                                                path='miovision_results/resnet50_mio/resnet50_mio_uniform_class_checkpoint.pt')
            elif config_parser.model_name == 'vgg16':
                optimizer = torch.optim.Adam(params=optimize_parameters, lr=config_parser.learning_rate, weight_decay=config_parser.weight_decay)
                # initialize the early_stopping object
                early_stopping = EarlyStopping(patience_epochs=config_parser.patience, verbose=True, 
                                                path='miovision_results/vgg16_mio/vgg16_mio_checkpoint.pt')
            for epoch in range(config_parser.num_epochs):
                ###################
                # train the model #
                ###################
                model.train()
                learning_rate = config_parser.learning_rate
                if epoch == 12:
                    learning_rate /= 5
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rate
                if epoch == 20:
                    learning_rate /= 5
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rate
                
                for i, x in enumerate(train_loader):
                    images = Variable(x['image']).to(device)
                    labels = Variable(x['label']).to(device)
                    outputs = model(images.float())
                    train_loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    train_losses.append(train_loss.item())
                    if i % 50 == 0:
                        print('Epoch:', (epoch + 1), 'Train Batch:', i, ' Loss:', train_loss.item())

                ######################
                # validate the model #
                ######################
                model.eval()
                for i, x in enumerate(validation_loader):
                    val_images = Variable(x['image']).to(device)
                    val_labels = Variable(x['label']).to(device)
                    val_outputs = model(val_images.float())
                    val_loss = criterion(val_outputs, val_labels)
                    _, val_accuracy = accuracy(val_outputs, val_labels)
                    validation_accuracies.append(val_accuracy)
                    valid_losses.append(val_loss.item())
                    if i % 50 == 0:
                        print('Epoch:', (epoch + 1), 'Val Batch:', i, ' Loss:', val_loss.item())

                avg_train_losses = np.average(train_losses)
                avg_valid_losses = np.average(valid_losses)
                print('Epoch [{}/{}], Train_Loss: {:.4f}, Val_Loss: {:.4f}, Val_accuracy: {:.4f}'
                    .format(epoch + 1, config_parser.num_epochs, avg_train_losses, avg_valid_losses, np.average(validation_accuracies)))
                # clear lists to track next epoch
                train_losses = []
                valid_losses = []

                # early_stopping needs the validation loss to check if it has decreased,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(avg_valid_losses, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        else:
            # load the last checkpoint with the best model
            if config_parser.model_name == 'resnet50':
                model = resnet50_load_checkpoint(model)            
            elif config_parser.model_name == 'vgg16':
                model = vgg16_load_checkpoint(model)
            model.eval()
            if config_parser.separate_correct_val:
                correct_val_df = pd.DataFrame(columns=['File_name', 'Class'])
            for i, x in enumerate(validation_loader):
                with torch.no_grad():
                    val_images = Variable(x['image']).to(device)
                    val_labels = Variable(x['label']).to(device)
                    val_outputs = model(val_images.float())
                val_output_flags, val_batch_accuracy = accuracy(val_outputs, val_labels)
                val_accuracies.append(val_batch_accuracy)
                print('Validation Batch :', i, ' Accuracy:', val_batch_accuracy)
                if config_parser.separate_correct_val:
                    image_paths = x['image_path']
                    correct_val_df = separate_correctly_classified_images(val_output_flags, image_paths, correct_val_df, config_parser)
            print('Validation accuracy average:', np.average(val_accuracies))
            if config_parser.separate_correct_val:
                correct_val_csv_path = config_parser.root_directory + 'correct_val_mapping_' + config_parser.model_name + '.csv'
                correct_val_df.to_csv(correct_val_csv_path, index=False, header=None)
    # to evaluate the model with test data without having labels
    else:
        _, _, test_loader = data_loader(parser=config_parser)    
        model = resnet50_load_checkpoint(model)
        model.eval()
        for i, x in enumerate(test_loader):
            test_images = Variable(x['image']).to(device)
            test_outputs = model(test_images.float())
            _, test_prediction = test_outputs.topk(k=1, dim=1, largest=True, sorted=True)
            test_prediction_labels = []
            test_prediction_index = test_prediction.t().cpu().numpy()[0, :]
            for k in test_prediction_index:
                test_prediction_labels.append(get_label_for_testing[k])
            print(test_prediction_labels)


if __name__ == "__main__":
    main()