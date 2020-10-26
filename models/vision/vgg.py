import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as tmodels

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

from models.utils.inspect_model import count_parameters, plot_lr_finder, train_eval, epoch_time
from models.utils.lr_finder import LRFinder
from models.vision.visionbase import VisionBase
from models.modelbase import ModelBase


class VGG(torch.nn.Module, VisionBase, ModelBase):

    vgg_config = {
        'vgg11': [64, 'M',
                  128, 'M',
                  256, 256, 'M',
                  512, 512, 'M',
                  512, 512, 'M'],
        'vgg13': [64, 64, 'M',
                  128, 128, 'M',
                  256, 256, 'M',
                  512, 512, 'M',
                  512, 512, 'M'],
        'vgg16': [64, 64, 'M',
                  128, 128, 'M',
                  256, 256, 256, 'M',
                  512, 512, 512, 'M',
                  512, 512, 512, 'M'],
        'vgg17': [64, 64, 'M',
                  128, 128, 'M',
                  256, 256, 256, 256, 'M',
                  512, 512, 512, 512, 'M',
                  512, 512, 512, 512, 'M']
    }

    def __init__(self,
                 output_dim,
                 config_key='vgg11',
                 version="vgg_mnist",
                 batch_norm=False,
                 freeze_features=False,
                 classifier_size=7):
        super().__init__()

        self.config_key = config_key
        self.batch_norm = batch_norm
        self.output_dim = output_dim

        self.classifier_size = classifier_size

        self.features = VGG.get_vgg_layers(config_key, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d(self.classifier_size)

        self.classifier = nn.Sequential(
            nn.Linear(512 * pow(self.classifier_size, 2), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)  # CNN
        x = self.avgpool(x)  # standartize the size
        h = x.view(x.shape[0], -1)  # straighten
        x = self.classifier(h)  # classify

        return x, h

    def set_features(self, *argv, **kwargs):
        if len(argv) > 0:
            self.config_key = argv[0]
        elif 'config_key' in kwargs:
            self.config_key = kwargs['config_key']
        if len(argv) > 1:
            self.batch_norm = argv[1]
        elif 'batch_norm' in kwargs:
            self.batch_norm = kwargs['batch_norm']

        self.features = VGG.get_vgg_layers(*argv, **kwargs)

    def count_parameters(self):
        return count_parameters(self)

    @staticmethod
    def get_vgg_layers(config_key, batch_norm,
                       norm_after_activation=False,
                       in_channels=3):
        if config_key not in VGG.vgg_config:
            raise Exception("config_key %s not known" % config_key)

        layers = []
        for c in VGG.vgg_config[config_key]:
            if c == "M":
                layers.append(nn.MaxPool2d(kernel_size=2))  # stride==kernel_size
            else:
                layers.append(nn.Conv2d(in_channels, c, kernel_size=3, padding=1))

                layers_relu = [nn.ReLU(inplace=True)]
                if batch_norm:
                    layers_relu.append(nn.BatchNorm2d(c))
                    if not norm_after_activation:
                        layers_relu.reverse()
                layers += layers_relu
                in_channels = c

        return nn.Sequential(*layers)

    def freeze_features(self, unfreeze=False):
        for parameter in self.features.parameters():
            parameter.requires_grad = unfreeze

    def freeze_classifier(self, unfreeze=False):
        for parameter in self.classifier[:-1].parameters():
            parameter.requires_grad = unfreeze

    def set_transforms(self, size=None, means=None, stds=None):
        size = self.norm_size if size is None else size
        means = self.norm_means if means is None else means
        stds = self.norm_stds if stds is None else stds

        self.train_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(size, padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

    def prepare_dataloaders(self, batch_size):
        self.train_dataloader = data.DataLoader(
            self.train_data, shuffle=True, batch_size=batch_size)

        self.valid_dataloader = data.DataLoader(
            self.valid_data, batch_size=batch_size)

        self.test_dataloader = data.DataLoader(
            self.test_data, batch_size=batch_size)

    def split_data(self, valid_ratio, scale_down):
        n_train_examples = int(len(self.train_data) * valid_ratio)
        n_valid_examples = len(self.train_data) - n_train_examples

        # scale down the test, valid and train samples
        if scale_down is not None:
            self.test_data, _ = data.random_split(
                self.test_data, [int(len(self.test_data) * scale_down),
                                 int(len(self.test_data) * (1 - scale_down))])

            self.train_data, self.valid_data, _ = data.random_split(
                self.train_data,
                [int(n_train_examples * scale_down),
                 int(n_valid_examples * scale_down),
                 int(len(self.train_data) * (1 - scale_down))])

        else:
            self.train_data, self.valid_data = data.random_split(
                self.train_data, [n_train_examples, n_valid_examples])

        # Use the test_transforms on valid set as well
        self.valid_data = copy.deepcopy(self.valid_data)
        self.valid_data.dataset.transform = self.test_transforms

        print(f'Number of training examples: {len(self.train_data)}\n' \
              f'Number of validation examples: {len(self.valid_data)}\n' \
              f'Number of testing examples: {len(self.test_data)}')

    def load_data(self, dataset_name='CIFAR10'):

        self.train_data = getattr(datasets, dataset_name)(
            '.data',
            train=True,
            download=True,
            transform=self.train_transforms)

        self.test_data = getattr(datasets, dataset_name)(
            '.data',
            train=False,
            download=True,
            transform=self.test_transforms)

        self.classes = self.test_data.classes

    def prep_data(self, dataset_name='CIFAR10', valid_ratio=0.9,
                  batch_size=128, scale_down=None):

        self.batch_size = batch_size

        self.load_data(dataset_name)

        self.split_data(valid_ratio, scale_down)

        self.prepare_dataloaders(self.batch_size)

    def setup_training(self, lr=1e-7, optimizer='Adam', pretrain=None):
        if pretrain is None:
            pretrain = self.pretrain

        params = [
            {'params': self.features.parameters(),
             'lr': lr / 10 if pretrain else lr},
            {'params': self.classifier.parameters()}
        ]

        self.optimizer = getattr(optim, optimizer)(params, lr=lr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss = nn.CrossEntropyLoss()

        self = self.to(self.device)
        self.loss = self.loss.to(self.device)

    def find_lr(self, end_lr=10, num_iter=100):

        lr_finder = LRFinder(self, self.optimizer, self.loss, self.device)
        lrs, losses = lr_finder.range_test(self.train_dataloader, end_lr, num_iter)

        plot_lr_finder(lrs, losses, skip_start=10, skip_end=20)

        return lrs[losses.index(min(losses))], min(losses)

    def init_weights(self, pretrain=True):
        if pretrain:
            attribute = self.config_key + self.batch_norm * ('_bn')
            pretrained_model = getattr(tmodels, attribute)(pretrained=True)

            # replace last layer if outputs don't match in dim
            if pretrained_model.classifier[-1].out_features != self.output_dim:
                pretrained_model.classifier[-1] = nn.Linear(
                    pretrained_model.classifier[-1].in_features,
                    self.output_dim)

            # from pytorch docs
            self.pretrain = True
            self.norm_size = 224
            self.norm_means = [0.485, 0.456, 0.406]
            self.norm_stds = [0.229, 0.224, 0.225]

            # load the weights
            self.load_state_dict(pretrained_model.state_dict())
        else:
            pass

    def do_train(self, epochs=5):
        best_valid_loss = float('inf')

        for epoch in range(epochs):

            start_time = time.monotonic()

            train_loss, train_acc = train_eval(
                self, self.train_dataloader, self.optimizer,
                self.loss, self.device, mode='train')

            valid_loss, valid_acc = train_eval(
                self, self.valid_dataloader, None,
                self.loss, self.device, mode='eval')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'tut4-model.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
