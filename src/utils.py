# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Ultility classes and functions.
"""
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt


class ImageData:
    """Image data used for classification"""
    def __init__(self,
                 images,
                 image_shape,
                 labels=None):
        # pass in the data
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.num_images = self.images.shape[0]

        # organize labels
        if self.labels is None:
            self.unique_labels = None
            self.num_classes = None
            self.class_sizes = None
            self.class_slices = None
        else:
            (self.unique_labels,
             self.class_sizes) = np.unique(self.labels, return_counts=True)
            self.num_classes = self.unique_labels.size
            self.class_slices = sizes_to_slices(self.class_sizes)

            sort_id = np.argsort(self.labels)
            self.images = self.images[sort_id]
            self.labels = self.labels[sort_id]

    def plot_image(self, image_id):
        """plot image data for given image_id"""
        image = self.images[image_id]
        image = image.reshape(self.image_shape)
        plt.imshow(image)


class BinaryImageClassifier:
    """Result from logistic regression"""
    def __init__(self,
                 classifier,
                 image_shape,
                 class_labels=np.array([-1, 1])):
        # pass in the data
        self.classifier = classifier
        self.image_shape = image_shape
        self.class_labels = class_labels

    def modify_class_labels(self, class_labels):
        """change to other labels"""
        self.class_labels = class_labels

    def classify_images(self, images):
        """predict the given image(s)"""
        if images.ndim == 1:
            images = images.reshape(1, images.size)
        num_images = images.shape[0]

        pred = images.dot(self.classifier)
        class0_id = pred < 0.0
        class1_id = pred >= 0.0

        labels = np.empty(num_images, dtype=self.class_labels.dtype)
        labels[class0_id] = self.class_labels[0]
        labels[class1_id] = self.class_labels[1]

        return labels

    def plot_classifier(self):
        """plot the classifier for fun"""
        classifier = self.classifier.reshape(self.image_shape)
        plt.imshow(classifier)


def sizes_to_slices(sizes):
    """convert sizes to slices"""
    slices = []
    break_points = np.cumsum(np.insert(sizes, 0, 0))
    for i in range(len(sizes)):
        slices.append(slice(break_points[i], break_points[i + 1]))
    return slices


def project_onto_capped_simplex(w, w_sum):
    """project onto the capped simplex"""
    a = np.min(w) - 1.0
    b = np.max(w) - 0.0

    def f(x):
        return np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - w_sum

    x = spopt.bisect(f, a, b)

    return np.maximum(np.minimum(w - x, 1.0), 0.0)
