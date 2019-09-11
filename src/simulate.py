# -*- coding: utf-8 -*-
"""
    simulate
    ~~~~~~~~

    Random simulation of the images for classification.
"""
import numpy as np
import utils


def simulate_images(image_shape, class_sizes):
    """generate data used for logistic regression"""
    rectangles = generate_rectangles(image_shape, class_sizes[0])
    crosses = generate_crosses(image_shape, class_sizes[1])
    windows = generate_windows(image_shape, class_sizes[2])

    images = np.vstack((rectangles, crosses, windows))

    labels = np.repeat([0, 1], class_sizes[:2])
    labels = np.hstack((labels,
                        np.random.choice(range(2), class_sizes[2])))

    return utils.ImageData(images, image_shape, labels=labels)


def generate_rectangles(image_shape, num_images):
    """randomly generate rectangles"""
    if num_images == 0:
        return np.array([]).reshape(0, np.prod(image_shape))
    images = []
    for i in range(num_images):
        images.append(_generate_rectangle(image_shape))

    return np.vstack(images)


def _generate_rectangle(image_shape):
    """randomly genrate a single rectangle"""
    h_range_list, w_range_list = shape_to_ranges(image_shape)

    a = [np.random.choice(h_range_list[0]),
         np.random.choice(h_range_list[2])]
    b = [np.random.choice(w_range_list[0]),
         np.random.choice(w_range_list[2])]

    image = np.zeros(image_shape)
    # draw four lines
    image[a[0], b[0]:(b[1] + 1)] = 1.0
    image[a[1], b[0]:(b[1] + 1)] = 1.0
    image[a[0]:(a[1] + 1), b[0]] = 1.0
    image[a[0]:(a[1] + 1), b[1]] = 1.0

    return image.reshape(image.size,)


def generate_crosses(image_shape, num_images):
    """randomly generate crosses"""
    if num_images == 0:
        return np.array([]).reshape(0, np.prod(image_shape))
    images = []
    for i in range(num_images):
        images.append(_generate_cross(image_shape))

    return np.vstack(images)


def _generate_cross(image_shape):
    """randomly generate a single cross"""
    h_range_list, w_range_list = shape_to_ranges(image_shape)

    x = np.random.choice(h_range_list[1])
    y = np.random.choice(w_range_list[1])
    a = [np.random.choice(h_range_list[0]),
         np.random.choice(h_range_list[2])]
    b = [np.random.choice(w_range_list[0]),
         np.random.choice(w_range_list[2])]

    image = np.zeros(image_shape)
    # draw two lines
    image[x, b[0]:(b[1] + 1)] = 1.0
    image[a[0]:(a[1] + 1), y] = 1.0

    return image.reshape(image.size,)


def generate_windows(image_shape, num_images):
    """randomly generate windows used as outliers"""
    if num_images == 0:
        return np.array([]).reshape(0, np.prod(image_shape))
    images = []
    for i in range(num_images):
        images.append(_generate_window(image_shape))

    return np.vstack(images)


def _generate_window(image_shape):
    """randomly gnerate a single window"""
    h_range_list, w_range_list = shape_to_ranges(image_shape)

    x = np.random.choice(h_range_list[1])
    y = np.random.choice(w_range_list[1])
    a = [np.random.choice(h_range_list[0]),
         np.random.choice(h_range_list[2])]
    b = [np.random.choice(w_range_list[0]),
         np.random.choice(w_range_list[2])]

    image = np.zeros(image_shape)
    # draw six lines
    image[a[0], b[0]:(b[1] + 1)] = 1.0
    image[a[1], b[0]:(b[1] + 1)] = 1.0
    image[a[0]:(a[1] + 1), b[0]] = 1.0
    image[a[0]:(a[1] + 1), b[1]] = 1.0

    image[x, b[0]:(b[1] + 1)] = 1.0
    image[a[0]:(a[1] + 1), y] = 1.0

    return image.reshape(image.size,)


def shape_to_ranges(image_shape):
    """convert shape to ranges"""
    h, w = image_shape
    h_break_points = [0, int(h/3), int(2*h/3), h]
    w_break_points = [0, int(w/3), int(2*w/3), w]

    h_range_list = []
    w_range_list = []
    for i in range(3):
        h_range_list.append(range(h_break_points[i], h_break_points[i + 1]))
        w_range_list.append(range(w_break_points[i], w_break_points[i + 1]))

    return h_range_list, w_range_list
