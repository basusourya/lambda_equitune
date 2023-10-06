import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False,
                compute_distance=False, phases=['train', 'val'], batch_size=1):
    since = time.time()

    val_acc_history = []
    equitune_dist_hist = []
    model.use_e_loss = False
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if compute_distance:
                # stores the norm between original features and equitune features
                running_equitune_dist = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        raise "equi-tuning does not currently support inception model, please try one of resnet," \
                              " alexnet, vgg, or densenet"
                        # outputs, aux_outputs = model(inputs)
                        # loss1 = criterion(outputs, labels)
                        # loss2 = criterion(aux_outputs, labels)
                        # loss = loss1 + 0.4 * loss2
                    else:
                        if model.use_e_loss:
                            outputs = model(inputs)
                            x_n, weights_n = outputs
                            weights = weights_n.view(4, batch_size, -1)
                            x = x_n.view(4, batch_size, -1)
                            outputs = nn.functional.softmax(x, dim=2)
                            loss_n = torch.stack([-torch.log(outputs[i, range(outputs.shape[1]), labels]) for i in range(4)], dim=0)
                            loss_k = torch.sum(loss_n*weights[:, :, 0], dim=0)
                            loss = torch.mean(loss_k)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    if model.use_e_loss:
                        _, preds = torch.max(weights, 0)
                        x_preds = x.permute(1, 0, 2)
                        #print (x_preds.size())
                        argmax_predictions = x_preds[range(batch_size), preds.view(-1), :]
                        #print (preds.size(), argmax_predictions.size())
                        _, preds = torch.max(argmax_predictions, 1)

                        
                    else:
                        _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if not compute_distance:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            else:
                epoch_equitune_dist = running_equitune_dist / len(dataloaders[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f} Equitune Distance: {:.4f}'.format(phase, epoch_loss,
                                                                                     epoch_acc, epoch_equitune_dist))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                print('Best val Acc: {:4f}'.format(best_acc))
                if compute_distance:
                    equitune_dist_hist.append(epoch_equitune_dist)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, best_acc


def eval_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False,
                compute_distance=False):
    since = time.time()

    val_acc_history = []
    equitune_dist_hist = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
            model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                print('Best val Acc: {:4f}'.format(best_acc))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_acc

