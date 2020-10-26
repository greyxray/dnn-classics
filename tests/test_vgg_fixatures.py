'''
    To run locally : python -m pytest -v -r s -s
'''
import pytest

from models.utils.random_state import RandomState
from models.vision.vgg import VGG
from models.utils.inspect_model import count_parameters

import copy


@pytest.fixture
def random_state(state=137):
    random_state_ = RandomState(state)
    random_state_.fix()
    return state


@pytest.fixture
def vgg_nopretrain():
    model = VGG(output_dim=10, config_key='vgg11')

    # No pretrainings available
    model.set_features(
        config_key='vgg11',
        batch_norm=True,
        norm_after_activation=True,
    )

    return model


def test_vgg_count_parameters_nopretrain(random_state, vgg_nopretrain):
    assert count_parameters(vgg_nopretrain) == 128812810

    vgg_nopretrain.freeze_features()
    assert count_parameters(vgg_nopretrain) == 119586826

    vgg_nopretrain.freeze_classifier()
    assert count_parameters(vgg_nopretrain) == 40970


@pytest.fixture
def vgg():
    model = VGG(output_dim=10, config_key='vgg11')

    return model


def test_vgg_count_parameters(random_state, vgg):
    assert count_parameters(vgg) == 128807306

    vgg.freeze_features()
    assert count_parameters(vgg) == 119586826

    vgg.freeze_classifier()
    assert count_parameters(vgg) == 40970

    vgg.freeze_features(unfreeze=True)
    assert count_parameters(vgg) == 9261450

    vgg.freeze_classifier(unfreeze=True)
    assert count_parameters(vgg) == 128807306
