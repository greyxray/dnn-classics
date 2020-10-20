#!/usr/bin/env python
import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from models.utils.random_state import RandomState
from models.vision.vgg import VGG
from models.utils.inspect_model import count_parameters
from models.utils.inspect_image import plot_train_images


def main():

    random_state = RandomState(137)
    random_state.fix()

    model = VGG(output_dim=10, config_key='vgg11')

    # # No pretrainings available
    # model.set_features(
    #     config_key='vgg11',
    #     batch_norm=True, norm_after_activation=True)

    model.init_weights(pretrain=True)

    # print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # model.freeze_features()
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    # model.freeze_classifier()
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    model.set_transforms()

    model.load_data(dataset_name='CIFAR10')

    # model.plot_train_images(5)

    model.setup_training()

    lr, loss = model.find_lr()
    print(lr, loss)

    print("done")


if __name__ == "__main__":
    main()
