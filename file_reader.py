import numpy as np
import matplotlib.pyplot as plt

NORMALIZE_FACTOR = 0.99 / 255
NUMBER_OF_LABELS = 10


def open_file(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


# GET ONLY WHAT'S NEXT AFTER THE FIRST VALUE OF ARRAY
# NORMALIZE IN ORDER TO AVOID 0 VALUES => PREVENT NO WEIGHT UPDATE
def normalize_data(data):
    return np.asfarray(data[:,1:]) * NORMALIZE_FACTOR + 0.01


# GET ONLY THE LABEL VALUE (FIRST VALUE)
def normalize_labels(data):
    return np.asfarray(data[:,:1])


# TRANSFORM LABELS FROM INITIAL FORM TO ONE HOT FORM(ARRAY OF 0 AND 1)
def label_to_one_hot(labels):
    labels_array = np.arange(NUMBER_OF_LABELS)
    labels_array = (labels_array == labels).astype(np.float)
    labels_array[labels_array == 0] = 0.01
    labels_array[labels_array == 1] = 0.99
    return labels_array


# OPEN IMAGE
def open_image(image):
    image = image.reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.show()




