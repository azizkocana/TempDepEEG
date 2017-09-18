""" Demo for ConditionalVariationalAutoencoder class
    uses mnist digit data set """
import numpy as np
import matplotlib as mpl

mpl.use('Qt4Agg')

import matplotlib.pyplot as plt
from scipy.stats import norm
from model_lib import ConditionalVariationalAutoEncoder

from keras.utils import to_categorical
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
class_num = 10
intermediate_dim = 256
epochs = 20
num_layers = 2
epsilon_std = 1.0

c_vae = ConditionalVariationalAutoEncoder(x_dim=original_dim,
                                          c_dim=class_num,
                                          h_dim=intermediate_dim,
                                          z_dim=latent_dim,
                                          num_layers=num_layers,
                                          batch_size=batch_size,
                                          max_num_iter=epochs)

# train the CVAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
cy_train = to_categorical(y_train)
cy_test = to_categorical(y_test)

c_vae.fit_c_vae(train_x=[x_train, cy_train], train_y=x_train,
                val_x=[x_test, cy_test], val_y=x_test)

# build a model to project inputs on the latent space
encoder = c_vae.encoder

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict([x_test, cy_test], batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
generator = c_vae.generator

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

y_sample = to_categorical(np.asarray(
    [0] * 15 + [1] * 15 + [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [
        6] * 15 + [7] * 15 + [8] * 15 + [9] * 15 + [1] * 15 + [2] * 15 + [
        3] * 15 + [4] * 15 + [5] * 15))

# TODO: to generate you have to pass an array of arrays\
# Check the data structure
counter = 0
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([xi, yi])
        print(z_sample)
        gen = np.array([np.concatenate((z_sample, y_sample[counter, :]))])
        x_decoded = generator.predict(gen)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit
        counter += 1

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
