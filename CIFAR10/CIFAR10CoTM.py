from PyCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from keras.datasets import cifar10
import cv2

clauses = 80000
T = 15000
s = 20.0
patch_size = 8
resolution = 8
epochs = 30

def horizontal_flip(image):
    return cv2.flip(image, 1)

augmented_images = []
augmented_labels = []

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

for i in range(len(X_train_org)):
    image = X_train_org[i]
    label = Y_train[i]

    # Original image and label
    augmented_images.append(image)
    augmented_labels.append(label)

augmented_images.append(horizontal_flip(image))
augmented_labels.append(label)

X_train_aug = np.array(augmented_images)
Y_train = np.array(augmented_labels).reshape(-1, 1)

X_train = np.copy(X_train_aug)
X_test = np.copy(X_test_org)

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[3]):
        X_train[i, :, :, j] = cv2.adaptiveThreshold(
            X_train_aug[i, :, :, j],
            1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[3]):
        X_test[i, :, :, j] = cv2.adaptiveThreshold(
            X_test_org[i, :, :, j],
            1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

f = open("cifar10_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")


tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), q=1.0, number_of_state_bits=8)

for i in range(epochs):
        start_training = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100*(tm.predict(X_train) == Y_train).mean()
        print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
        print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
        f.flush()
    
f.close()