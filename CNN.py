import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class Conv_op:

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def image_region(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[j: (j + self.filter_size), k:(k + self.filter_size)]
                yield image_patch, j, k

    def forward_prop(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_patch, i, j in self.image_region(image):
            conv_out[i, j] = np.sum(image_patch * self.conv_filter, axis=(1, 2))
        return conv_out

    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)

        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] += image_patch * dL_dout[i, j, k]

        # Filter params update
        self.conv_filter -= learning_rate * dL_dF_params
        return dL_dF_params


class Max_Pool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i * self.filter_size):(i * self.filter_size + self.filter_size),
                              (j * self.filter_size):(j * self.filter_size + self.filter_size)]
                yield image_patch, i, j

    def forward_prop(self, image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))
        for image_patch, i, j in self.image_region(image):
            output[i, j] = np.amax(image_patch, axis=(0, 1))
        return output

    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis=(0, 1))

            for il in range(height):
                for jl in range(width):
                    for kl in range(num_filters):
                        if image_patch[il, jl, kl] == maximum_val[kl]:
                            dL_dmax_pool[i * self.filter_size + il, j * self.filter_size + jl, kl] = dL_dout[i, j, kl]
            return dL_dmax_pool


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)

    def forward_prop(self, image):
        self.orig_im_shape = image.shape  # used in backprop
        image_modified = image.flatten()
        self.modified_input = image_modified  # to be used in backprop
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out / np.sum(exp_out, axis=0)

    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue

            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)
            # Gradients with respect to out(z)
            dy_dz = -transformation_eq[i] * transformation_eq / (S_total ** 2)
            dy_dz[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)
            # Gradients of totals against weights/biases/input
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight
            # Gradients of loss against totals
            dL_dz = grad * dy_dz
            # Gradients of loss against weights/biases/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz

            # Update weights and biases
            self.weight -= learning_rate * dL_dw
            self.bias == learning_rate * dL_db

            return dL_d_inp.reshape(self.orig_im_shape)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_images = X_train[:1500]
train_labels = y_train[:1500]
test_images = X_test[:1500]
test_labels = y_test[:1500]

conv = Conv_op(8, 3)
pool = Max_Pool(2)
softmax = Softmax(13 * 13 * 8, 10)


# 28x28x1->26x26x8
# 26x26x8->13x13x8
# 13x13x8->10 138,10)


def cnn_forward_prop(image, label):
    out_p = conv.forward_prop((image / 255) - 0.5)
    out_p = pool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)

    # Calculate cross-entropy loss and accuracy.
    cross_ent_loss = -np.log(out_p[label])
    accuracy_eval = 1 if np.argmax(out_p) == label else 0

    return out_p, cross_ent_loss, accuracy_eval


def training_cnn(image, label, learn_rate=.005):
    # Forward
    out, loss, acc = cnn_forward_prop(image, label)
    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    # Backprop
    grad_back = softmax.back_prop(gradient, learn_rate)
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learn_rate)
    return loss, acc


for epochl in range(4):
    print('Epoch%d--->' % (epochl + 1))

    # Shuffle the training data
    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]
    # Training the CNN
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 0:
            print('%d steps out of 100 steps: Average Loss %.3f and Accuracy:%d%%' % (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0
        l1, accu = training_cnn(im, label)
        loss += l1
        num_correct += accu

# Testing the CNN
print('** Testing phase')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l1, accu = cnn_forward_prop(im, label)
    loss += l1
    num_correct += accu

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

