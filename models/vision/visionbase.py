import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.modelbase import ModelBase


class VisionBase(ModelBase):

    @staticmethod
    def normalize_image(image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min=image_min, max=image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image

    @staticmethod
    def get_incorrect(labels, pred_labels, images, probs):
        corrects = torch.eq(labels, pred_labels)

        incorrect_examples = []

        for image, label, prob, correct in zip(images, labels, probs, corrects):
            if not correct:
                incorrect_examples.append((image, label, prob))

        incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

        return incorrect_examples

    @staticmethod
    def plot_images(images, labels, classes, normalize=True):

        n_images = len(images)

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=(10, 10))

        for i in range(rows * cols):

            ax = fig.add_subplot(rows, cols, i + 1)

            image = images[i]

            if normalize:
                image = VisionBase.normalize_image(image)

            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            ax.set_title(classes[labels[i]])
            ax.axis('off')

        plt.show()

    @staticmethod
    def plot_images_data(data, classes, n_images=25):
        images, labels = zip(*[(image, label) for image, label in
                               [data[i] for i in range(n_images)]])

        # classes = test_data.classes  # not sure why the train data has no classes

        VisionBase.plot_images(images, labels, classes)

    def plot_train_images(self, n_images=25):
        VisionBase.plot_images_data(
            data=self.train_data,
            classes=self.classes,
            n_images=n_images)

    # Plotting the most confident incorrect predictions
    @staticmethod
    def plot_most_incorrect(incorrect, classes, n_images, normalize=True, show=True):

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=(25, 20))

        for i in range(rows * cols):

            ax = fig.add_subplot(rows, cols, i + 1)

            image, true_label, probs = incorrect[i]
            image = image.permute(1, 2, 0)
            true_prob = probs[true_label]
            incorrect_prob, incorrect_label = torch.max(probs, dim=0)
            true_class = classes[true_label]
            incorrect_class = classes[incorrect_label]

            if normalize:
                image = VisionBase.normalize_image(image)

            ax.imshow(image.cpu().numpy())
            ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n' \
                         f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
            ax.axis('off')

        fig.subplots_adjust(hspace=0.4)

        if show:
            plt.show()

    def plot_test_most_incorrect(self, classes=None, n_images=36, normalize=True, show=True):
        incorrect = VisionBase.get_incorrect(self.test_labels, self.test_pred_labels, self.test_images, self.test_probs)

        if classes is None:
            classes = self.classes

        VisionBase.plot_most_incorrect(incorrect, classes, n_images, normalize, show)

    # Collect the predictions
    @staticmethod
    def get_image_predictions(model, dataloader, device):
        '''
        Get predictions on the images
        '''
        model.eval()

        images, labels, probs, max_preds = [], [], [], []

        with torch.no_grad():
            for (x, y) in dataloader:

                x = x.to(device)

                y_pred, h = model(x)

                y_prob = F.softmax(y_pred, dim=-1)
                max_pred = y_prob.argmax(1, keepdim=True)

                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())
                max_preds.append(max_pred.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        pred_labels = torch.argmax(probs, 1)

        return images, labels, probs, pred_labels, max_preds

    def get_predictions(self, sample, device=None):
        if device is None:
            device = self.device

        return VisionBase.get_image_predictions(self, sample, device=device)

    def get_test_predictions(self, device=None):

        self.test_images, self.test_labels, self.test_probs, self.test_pred_labels, \
            self.test_max_preds = self.get_predictions(self.test_dataloader, device)

        return self.test_pred_labels

    @staticmethod
    def plot_filtered_images(images, filters, n_filters=None, normalize=True, show=True):

        images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
        filters = filters.cpu()

        if n_filters is not None:
            filters = filters[:n_filters]

        n_images = images.shape[0]
        n_filters = filters.shape[0]

        filtered_images = F.conv2d(images, filters)

        fig = plt.figure(figsize=(30, 30))

        for i in range(n_images):

            image = images[i]

            if normalize:
                image = VisionBase.normalize_image(image)

            ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
            ax.imshow(image.permute(1, 2, 0).numpy())
            ax.set_title('Original')
            ax.axis('off')

            for j in range(n_filters):
                image = filtered_images[i][j]

                if normalize:
                    image = VisionBase.normalize_image(image)

                ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters) + j + 1)
                ax.imshow(image.numpy(), cmap='bone')
                ax.set_title(f'Filter {j+1}')
                ax.axis('off')

        fig.subplots_adjust(hspace=-0.7)

        if show:
            plt.show()

    def plot_n_filtered_images(self, n_images=5, n_filters=7, show=True):
        images = [image for image, label in [self.test_data[i] for i in range(n_images)]]
        filters = self.features[0].weight.data

        VisionBase.plot_filtered_images(images, filters, n_filters, show=show)

    @staticmethod
    def plot_filters(filters, normalize=True, show=True):

        filters = filters.cpu()

        n_filters = filters.shape[0]

        rows = int(np.sqrt(n_filters))
        cols = int(np.sqrt(n_filters))

        fig = plt.figure(figsize=(20, 10))

        for i in range(rows * cols):

            image = filters[i]

            if normalize:
                image = VisionBase.normalize_image(image)

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(image.permute(1, 2, 0))
            ax.axis('off')

        fig.subplots_adjust(wspace=-0.9)

        if show:
            plt.show()

    def plot_n_filters(self, n_filters=7, show=True):
        filters = self.features[0].weight.data

        VisionBase.plot_filters(filters, show=show)
