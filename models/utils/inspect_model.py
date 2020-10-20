#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import copy
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5):

    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()


def calculate_accuracy(y_pred, y):
    max_pred = y_pred.argmax(1, keepdim=True)
    correct = max_pred.eq(y.view_as(max_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_eval(model, iterator, optimizer, criterion, device, mode='eval'):

    epoch_loss = 0
    epoch_acc = 0

    if mode == 'train' and optimizer is None:
        raise Exception("Can't train with None optimizer")

    if mode == 'train':
        model.train()
    else:
        model.train()

    for (x, y) in iterator:
        print(y, mode, )
        x = x.to(device)
        y = y.to(device)


        if mode == 'train':
            print('optimizer zero grad')
            optimizer.zero_grad()

        print('pred', model)
        y_pred, _ = model(x)

        print('loss')
        loss = criterion(y_pred, y)

        print('acc')
        acc = calculate_accuracy(y_pred, y)


        if mode == 'train':
            print('backward')
            loss.backward()
            print('step')
            optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        print(epoch_loss)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
