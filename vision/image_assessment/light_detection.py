import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


class LightDetection:
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.labels = []
        self.preprocessed_imgs = []
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)

    @staticmethod
    def visualize_histograms(img, color_space="hsv"):
        # sns.set_theme()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bins = np.linspace(0, 256, 8)
        fig = plt.figure(figsize=(40, 5))
        plt.subplots_adjust(wspace=0.4)

        plt.subplot(1, 3, 1)
        plt.hist(hsv[:, :, 0].flatten(), bins, alpha=0.5, histtype='bar', ec='black', color='r', label='Hue')
        plt.title('Hue')
        plt.xlabel('Pixel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.hist(hsv[:, :, 1].flatten(), bins, alpha=0.5, histtype='bar', ec='black', color='g', label='Saturation')
        plt.title('Saturation')
        plt.xlabel('Pixel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.hist(hsv[:, :, 2].flatten(), bins, alpha=0.5, histtype='bar', ec='black', color='skyblue', label='Value')
        plt.title('Value')
        plt.xlabel('Pixel intensity')
        plt.ylabel('Frequency')
        plt.legend()

        plt.show()

    def preprocess(self):
        bins = np.linspace(0, 256, 8)
        for img in self.train_dir:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue_histogram = np.histogram(hsv[:, :, 0], bins)
            sat_histogram = np.histogram(hsv[:, :, 1], bins)
            val_histogram = np.histogram(hsv[:, :, 2], bins)
            combined_histograms = np.concatenate((hue_histogram, sat_histogram, val_histogram))
            self.preprocessed_imgs.append(combined_histograms)

    def fit(self):
        self.preprocess()
        self.model.fit(self.preprocessed_imgs, self.labels)

    def predict(self):
        pass


if __name__ == '__main__':
    img = cv2.imread('/Users/nishant/PycharmProjects/ArtificialIntelligence/data/familypic.jpeg')
    LightDetection.visualize_histograms(img)

