#!/usr/bin/env python
import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from models.utils.random_state import RandomState
from models.vision.vgg import VGG
from models.utils.inspect_model import count_parameters


def main():

    random_state = RandomState(137)
    random_state.fix()

    model = VGG(output_dim=10, config_key='vgg11')

    # # No pretrainings available
    # model.set_features(
    #     config_key='vgg11',
    #     batch_norm=True, norm_after_activation=True)

    model.init_weights(pretrain=True)
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.set_transforms()

    model.prep_data(
        dataset_name='CIFAR10',
        scale_down=0.01,
        batch_size=8)

    # model.plot_train_images(5)

    # Learning the right rate search
    # model.setup_training(pretrain=False, lr=1e-7)
    # lr, loss = model.find_lr()
    # print(lr, loss)

    model.setup_training(lr=5e-4)  # necessary for lr search

    # # Actual training : skipping
    # model.do_train(epochs=1)

    # Collect the predictions
    model.get_test_predictions()

    # Confusion matrix
    model.plot_test_confusion_matrix()

    # Plotting the most confident incorrect predictions
    model.plot_test_most_incorrect(n_images=36)

    # PCA and t-SNE
    model.get_train_representation()
    model.plot_pca_train_representation()
    model.plot_tsne_train_representation(n_images=20)

    # Images after convolutional layers
    model.plot_n_filtered_images(5, 7)

    # Values of filters
    model.plot_n_filters(5)

    print("done")


if __name__ == "__main__":
    main()
