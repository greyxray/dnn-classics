import torch
# from models.utils.inspect_image import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import manifold


class ModelBase(object):

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def plot_confusion_matrix(labels, pred_labels, classes, show=True):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        cm = confusion_matrix(labels, pred_labels)
        cm = ConfusionMatrixDisplay(cm, display_labels=classes)
        cm.plot(values_format='d', cmap='Blues', ax=ax)
        plt.xticks(rotation=20)

        if show:
            plt.show()

    def plot_test_confusion_matrix(self, show=True):
        ModelBase.plot_confusion_matrix(self.test_labels, self.test_pred_labels, self.classes, show=show)

    @staticmethod
    def get_tsne(data, n_components=2, n_images=None):

        if n_images is not None:
            data = data[:n_images]

        tsne = manifold.TSNE(n_components=n_components, random_state=0)
        tsne_data = tsne.fit_transform(data)
        return tsne_data

    @staticmethod
    def get_pca(data, n_components=2):
        pca = decomposition.PCA()
        pca.n_components = n_components
        pca_data = pca.fit_transform(data)
        return pca_data

    @staticmethod
    def plot_representations(data, labels, classes, n_images=None, show=True):

        if n_images is not None:
            data = data[:n_images]
            labels = labels[:n_images]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
        handles, labels = scatter.legend_elements()
        legend = ax.legend(handles=handles, labels=classes)

        if show:
            plt.show()

    @staticmethod
    def get_representations(model, iterator, device, intermediate=False):

        model.eval()

        outputs = []
        intermediates = []
        labels = []

        with torch.no_grad():

            for (x, y) in iterator:

                x = x.to(device)

                y_pred, h = model(x)

                outputs.append(y_pred.cpu())
                if intermediate:
                    intermediates.append(h.cpu())
                labels.append(y)

        outputs = torch.cat(outputs, dim=0)
        if intermediate:
            intermediates = torch.cat(intermediates, dim=0)
        labels = torch.cat(labels, dim=0)

        return outputs, intermediates, labels

    def get_train_representation(self, intermediate=False):
        self.train_outputs, self.train_h, self.train_labels = ModelBase.get_representations(self, self.train_dataloader, self.device, intermediate)
        return self.train_outputs, self.train_h, self.train_labels

    def plot_pca_train_representation(self, intermediate=False, show=True):
        output_pca_data = ModelBase.get_pca(self.train_outputs)
        ModelBase.plot_representations(output_pca_data, self.train_labels, self.classes, show=show)

    def plot_tsne_train_representation(self, n_images=10, intermediate=False, show=True):
        output_tsne_data = ModelBase.get_tsne(self.train_outputs, n_images=n_images)
        ModelBase.plot_representations(output_tsne_data, self.train_labels, self.classes, n_images, show=show)
