#!/usr/bin/env python
import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from models.utils.random_state import RandomState
from models.vision.resnet import ResNet
from models.utils.inspect_model import count_parameters


def main():

    random_state = RandomState(137)
    random_state.fix()

    # No need to call this every time
    # ResNet.prepare_train_test(train_ratio=0.8, data_dir='../data/CUB_200_2011')

    model = ResNet(config_key='resnet50', output_dim=200,
                   data_dir='../data/CUB_200_2011')

    model.init_weights(pretrain=True)
    print(model)

    model.set_transforms()

    model.prep_data()

    model.plot_train_images(5)

    # # Learning the right rate search
    # # model.setup_training(pretrain=False, lr=1e-7, discriminative_lr=False)
    # # lr, loss = model.find_lr()
    # # print(lr, loss)

    # optimized LR
    model.setup_training(lr=1e-3, discriminative_lr=True)

    # # # Actual training : skipping
    # # model.do_train(epochs=1)

    # # Collect the predictions
    # model.get_test_predictions()

    # # Confusion matrix
    # model.plot_test_confusion_matrix()

    # # Plotting the most confident incorrect predictions
    # model.plot_test_most_incorrect(n_images=36)

    # # PCA and t-SNE
    # model.get_train_representation()
    # model.plot_pca_train_representation()
    # model.plot_tsne_train_representation(n_images=20)

    # # Images after convolutional layers
    # model.plot_n_filtered_images(5, 7)

    # # Values of filters
    # model.plot_n_filters(5)

    print("done")


if __name__ == "__main__":
    main()
