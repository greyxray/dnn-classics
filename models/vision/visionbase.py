from models.utils.inspect_image import plot_train_images
from models.utils.inspect_model import get_image_predictions


class VisionBase(object):

    def plot_train_images(self, n_images=25):
        plot_train_images(
            data=self.train_data,
            classes=self.classes,
            n_images=n_images)


    def get_predictions(self, sample=None, device=None):
        if sample is None:
            sample = self.test_dataloader

        if device is None:
            device = self.device

        return get_image_predictions(self, sample, device=device)
