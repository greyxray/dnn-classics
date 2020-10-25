from models.utils.inspect_image import plot_confusion_matrix


class ModelBase(object):

    def plot_confusion_matrix(self, labels, pred_labels, classes=None):
        if classes is None:
            classes = self.classes

        plot_confusion_matrix(labels, pred_labels, classes)
