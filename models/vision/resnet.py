'''
Resnet with custom dataset and pretrained by pytorch model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import copy
from collections import namedtuple
import os
import shutil
import time

from models.utils.inspect_model import plot_lr_finder, epoch_time, train_eval_topk
from models.utils.lr_finder import LRFinder
from models.vision.visionbase import VisionBase


ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class BottleneckBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class ResNet(torch.nn.Module, VisionBase):

    resnet18_config = ResNetConfig(block=BasicBlock,
                                   n_blocks=[2, 2, 2, 2],
                                   channels=[64, 128, 256, 512])

    resnet34_config = ResNetConfig(block=BasicBlock,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    resnet50_config = ResNetConfig(block=BottleneckBlock,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    resnet101_config = ResNetConfig(block=BottleneckBlock,
                                    n_blocks=[3, 4, 23, 3],
                                    channels=[64, 128, 256, 512])

    resnet152_config = ResNetConfig(block=BottleneckBlock,
                                    n_blocks=[3, 8, 36, 3],
                                    channels=[64, 128, 256, 512])

    def __init__(self, config_key, output_dim, data_dir='../data/CUB_200_2011'):
        super().__init__()

        self.data_dir = data_dir
        self.config_key = config_key

        self.config = getattr(self, config_key + '_config')

        # self.batch_norm = batch_norm
        self.output_dim = output_dim

        block, n_blocks, channels = self.config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0], stride=1)
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h

    # def set_features(self, *argv, **kwargs):
    #     if len(argv) > 0:
    #         self.config_key = argv[0]
    #     elif 'config_key' in kwargs:
    #         self.config_key = kwargs['config_key']
    #     if len(argv) > 1:
    #         self.batch_norm = argv[1]
    #     elif 'batch_norm' in kwargs:
    #         self.batch_norm = kwargs['batch_norm']

    #     self.features = VGG.get_vgg_layers(*argv, **kwargs)

    @staticmethod
    def prepare_train_test(train_ratio=0.8, data_dir='../data/CUB_200_2011'):
        '''
            Prepare the directory structure

            TODO: instead of copy make softlinks
        '''
        images_dir = os.path.join(data_dir, 'images')
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        # remove train/test if exist
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        os.makedirs(train_dir)
        os.makedirs(test_dir)

        classes = os.listdir(images_dir)

        for c in classes:

            class_dir = os.path.join(images_dir, c)

            images = os.listdir(class_dir)

            n_train = int(len(images) * train_ratio)

            train_images = images[:n_train]
            test_images = images[n_train:]

            os.makedirs(os.path.join(train_dir, c), exist_ok=True)
            os.makedirs(os.path.join(test_dir, c), exist_ok=True)

            for image in train_images:
                image_src = os.path.join(class_dir, image)
                image_dst = os.path.join(train_dir, c, image)
                shutil.copyfile(image_src, image_dst)

            for image in test_images:
                image_src = os.path.join(class_dir, image)
                image_dst = os.path.join(test_dir, c, image)
                shutil.copyfile(image_src, image_dst)

    @staticmethod
    def get_mean_std_size(data_dir='../data/CUB_200_2011'):
        '''
            TODO: what if images have varying shape?
        '''
        train_dir = os.path.join(data_dir, 'train')

        # Transform images to tensors
        train_data = datasets.ImageFolder(root=train_dir,
                                          transform=transforms.ToTensor())

        means = torch.zeros(3)
        stds = torch.zeros(3)

        for img, label in train_data:
            # mean/std across the height and width dimensions with dim = (1,2)
            means += torch.mean(img, dim=(1, 2))
            stds += torch.std(img, dim=(1, 2))

        shape = img.shape[1]
        if img.shape[1] != img.shape[2]:
            shape = (img.shape[1], img.shape[2])
        means /= len(train_data)
        stds /= len(train_data)

        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')
        print(f'Image shape: {stds}')

        return means, stds, shape

    def load_data(self):

        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        self.train_data = datasets.ImageFolder(root=train_dir,
                                               transform=self.train_transforms)

        self.test_data = datasets.ImageFolder(root=test_dir,
                                              transform=self.test_transforms)

        # Shorten the classes names
        self.test_data.classes = [self.format_label(c) for c in self.test_data.classes]
        self.train_data.classes = [self.format_label(c) for c in self.train_data.classes]

        assert self.output_dim == len(self.test_data.classes)

        self.classes = self.test_data.classes

    def split_data(self, valid_ratio=0.9, scale_down=False):
        '''
            Split train dataset to train and valid

            TODO: scale_down
        '''
        n_train_examples = int(len(self.train_data) * valid_ratio)
        n_valid_examples = len(self.train_data) - n_train_examples

        self.train_data, valid_data = data.random_split(
            self.train_data,
            [n_train_examples, n_valid_examples])

        # transform validation as would the test set
        self.valid_data = copy.deepcopy(valid_data)
        self.valid_data.dataset.transform = self.test_transforms

        print(f'Number of training examples: {len(self.train_data)}')
        print(f'Number of validation examples: {len(self.valid_data)}')
        print(f'Number of testing examples: {len(self.test_data)}')

    def prepare_dataloaders(self, batch_size):
        self.train_dataloader = data.DataLoader(
            self.train_data, shuffle=True, batch_size=batch_size)

        self.valid_dataloader = data.DataLoader(
            self.valid_data, batch_size=batch_size)

        self.test_dataloader = data.DataLoader(
            self.test_data, batch_size=batch_size)

    def prep_data(self, normalize=True,
                  batch_size=64, valid_ratio=0.9, scale_down=False):

        self.batch_size = batch_size

        self.load_data()

        self.split_data(valid_ratio=valid_ratio, scale_down=scale_down)

        self.prepare_dataloaders(self.batch_size)

    def setup_training(self, lr=1e-7, discriminative_lr=False, optimizer='Adam'):
        '''
        Set ooptimizer, device and loss
        '''

        # set to False on the LR optimization
        if discriminative_lr:
            params = [
                {'params': self.conv1.parameters(), 'lr': lr / 10},
                {'params': self.bn1.parameters(), 'lr': lr / 10},
                {'params': self.layer1.parameters(), 'lr': lr / 8},
                {'params': self.layer2.parameters(), 'lr': lr / 6},
                {'params': self.layer3.parameters(), 'lr': lr / 4},
                {'params': self.layer4.parameters(), 'lr': lr / 2},
                {'params': self.fc.parameters()}
            ]
        else:
            params = [{'params': self.parameters()}]

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

            pretrained_model = getattr(models, self.config_key)(pretrained=True)
            print(pretrained_model)

            # create a new linear layer with the required dimensions.
            fc = nn.Linear(
                in_features=pretrained_model.fc.in_features,
                out_features=self.output_dim)
            pretrained_model.fc = fc

            '''
            # replace last layer if outputs don't match in dim
            if pretrained_model.classifier[-1].out_features != self.output_dim:
                pretrained_model.classifier[-1] = nn.Linear(
                    pretrained_model.classifier[-1].in_features,
                    self.output_dim)
            '''

            # ensure that our images are the same size and have the same
            # normalization as those used to train the model
            # from pytorch docs
            self.pretrain = True
            self.norm_size = 224
            self.norm_means = [0.485, 0.456, 0.406]
            self.norm_stds = [0.229, 0.224, 0.225]

            # load the pretrained weights
            self.load_state_dict(pretrained_model.state_dict())

        else:
            self.pretrain = False
            # TODO: check that tuple for norm_size is acceptable in transformer
            self.norm_means, self.norm_stds, self.norm_size = self.get_mean_std()

            # TODO: init weights randomly
            pass

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
            # No augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

    @staticmethod
    def format_label(label):
        label = label.split('.')[-1]
        label = label.replace('_', ' ')
        label = label.title()
        label = label.replace(' ', '')
        return label

    def do_train(self, epochs=10):

        # Prepare schedular
        steps_per_epoch = len(self.train_dataloader)
        tot_steps = epochs * steps_per_epoch

        max_lrs = [p['lr'] for p in self.optimizer.param_groups]
        scheduler = lr_scheduler.OneCycleLR(self.optimizer,
                                            max_lr=max_lrs,
                                            total_steps=tot_steps)

        # Train
        best_valid_loss = float('inf')

        for epoch in range(epochs):

            start_time = time.monotonic()

            train_loss, train_acc_1, train_acc_5 = train_eval_topk(
                model=self, iterator=self.train_dataloader,
                optimizer=self.optimizer, criterion=self.loss,
                scheduler=scheduler, device=self.device, mode='train')

            valid_loss, valid_acc_1, valid_acc_5 = train_eval_topk(
                model=self, iterator=self.valid_dataloader,
                optimizer=None, criterion=self.loss,
                scheduler=None, device=self.device, mode='eval')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'tut5-model.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
                  f'Train Acc @5: {train_acc_5*100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
                  f'Valid Acc @5: {valid_acc_5*100:6.2f}%')

        # Performance on test
        self.load_state_dict(torch.load('tut5-model.pt'))

        test_loss, test_acc_1, test_acc_5 = train_eval_topk(
            model=self, iterator=self.test_iterator,
            optimizer=None, criterion=self.loss,
            scheduler=None, device=self.device,  mode='eval')

        print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
              f'Test Acc @5: {test_acc_5*100:6.2f}%')
