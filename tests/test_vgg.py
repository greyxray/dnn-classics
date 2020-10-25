'''
    To run locally : python -m pytest -v -r s -s
'''
import unittest
import pytest

from models.utils.random_state import RandomState
from models.vision.vgg import VGG
from models.utils.inspect_model import count_parameters

import random
import numpy as np
import torch
import copy


def pytest_namespace():
    return {'model': None}


# @pytest.fixture(scope="session")
def set_random_numbers():
    random_state = RandomState(137)
    random_state.fix()


# @pytest.fixture(scope="session")
@pytest.mark.dependency()
def test_init_vgg():
    set_random_numbers()

    pytest.model = VGG(output_dim=10, config_key='vgg11')


@pytest.mark.dependency(depends=['test_init_vgg'])
def test_init_no_pretrain():
    # set_random_numbers()

    model = VGG(output_dim=10, config_key='vgg11')

    # # No pretrainings available
    model.set_features(
        config_key='vgg11',
        batch_norm=True, norm_after_activation=True)


@pytest.mark.dependency(depends=['test_init_vgg'])
def test_init_vgg_weight():

    pytest.model.init_weights(pretrain=True)


@pytest.mark.dependency(depends=['test_init_vgg_weight'])
def test_vgg_count_parameters():

    # print(f'\nThe model has {count_parameters(pytest.model):,} trainable parameters')
    assert count_parameters(pytest.model) == 128807306

    model = copy.deepcopy(pytest.model)
    model.freeze_features()
    # print(f'\nThe model has {count_parameters(model):,} trainable parameters')
    assert count_parameters(model) == 119586826

    model.freeze_classifier()
    # print(f'The model has {count_parameters(model):,} trainable parameters')
    assert count_parameters(model) == 40970
