import torch
import torch.nn as nn
import torch.optim as optim


class Net_trainer2:

    def __init__(self):
        self.dataset = []

    def train(self, net, trainloader, set_name, lrt, mom, nr_epochs):
        """
        Takes the dataset input in trainloader and the net architecture and trains the weights.
        Saves the weights to state folder.
        :param: net: Network nn.Module
        :param: trainloader: dataset, can be accessed via i, data in enumerate(trainloader, 0)
        :param: name of dataset as string, e.g. 'cifar', 'mnist'
        :param: learning rate, adapt here 0.001 for cifar, 0.01 for mnist
        :param: # nr_epochs: cifar 3, mnist 10?
        :return: PATH where the weights are stored
        """

        net.train() #set to training mode (turn off ranger)

        criterion = nn.CrossEntropyLoss() # define loss function
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #stochastic gradient descent, learning rate and momentum, cifar
        optimizer = optim.SGD(net.parameters(), lr=lrt, momentum=mom)  # stochastic gradient descent, learning rate and momentum, mnist, cifar

        print('training...')

        for epoch in range(nr_epochs):  # loop over the dataset multiple times
            print('epoch', epoch)
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0): #walk through all images, enumerate starting with zero

                inputs, labels = data

                optimizer.zero_grad() #reinitialize after each step()

                # forward + backward + optimize
                outputs = net(inputs) #Switch to false if no data should be collected

                loss = criterion(outputs, labels) #calculates loss
                loss.backward() #calculates the gradients (dloss/dweights)
                optimizer.step() #does parameter update

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        # Save the model
        PATH = str(set_name + '.pth')

        torch.save(net.state_dict(), PATH) #save weights
        print('weights saved')

        return PATH