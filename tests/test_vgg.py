'''
    To run locally : python -m pytest -v -r s -s
'''
import unittest
import pytest

from models.utils.random_state import RandomState
from models.vision.vgg import VGG
from models.utils.inspect_model import count_parameters

import copy


def pytest_namespace():
    return {'model': None}


def set_random_numbers():
    random_state = RandomState(137)
    random_state.fix()


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

    assert count_parameters(pytest.model) == 128807306

    model = copy.deepcopy(pytest.model)
    model.freeze_features()
    assert count_parameters(model) == 119586826

    model.freeze_classifier()
    assert count_parameters(model) == 40970


@pytest.mark.dependency(depends=['test_init_vgg_weight'])
def test_load_data():
    pytest.model.set_transforms()

    pytest.model.prep_data(
        dataset_name='CIFAR10',
        scale_down=0.005,
        batch_size=8)


@pytest.mark.dependency(depends=['test_load_data'])
def test_setup_training():
    pytest.model.setup_training(lr=5e-4)

# # Typically to expensive
# @pytest.mark.dependency(depends=['test_setup_training'])
# def test_training():
#     model.do_train(epochs=1)


@pytest.mark.dependency(depends=['test_setup_training'])
def test_predictions():
    pytest.model.get_test_predictions()


@pytest.mark.dependency(depends=['test_predictions'])
def test_plot_confusion_matrix():
    pytest.model.plot_test_confusion_matrix(show=False)


@pytest.mark.dependency(depends=['test_predictions'])
def test_plot_most_incorrect():
    pytest.model.plot_test_most_incorrect(n_images=4, show=False)


@pytest.mark.dependency(depends=['test_predictions'])
def test_get_train_representation():
    outputs, h, labels = pytest.model.get_train_representation()
    assert len(outputs) == len(labels)
    assert len(h) == 0

    pytest.model.plot_pca_train_representation(intermediate=False, show=False)
    pytest.model.plot_tsne_train_representation(n_images=10, intermediate=False, show=False)


@pytest.mark.dependency(depends=['test_predictions'])
def test_plot_n_filtered_images():
    pytest.model.plot_n_filtered_images(5, 7, show=False)


@pytest.mark.dependency(depends=['test_predictions'])
def test_plot_n_filters():
    pytest.model.plot_n_filters(7, show=False)
