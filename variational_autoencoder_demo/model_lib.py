""" This script includes models built using Keras with TensorFlow backend """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, LSTM, TimeDistributed
from keras.models import Model, Sequential
from keras.layers.merge import concatenate as concat
from keras import backend as K
from keras import metrics


def sampling(args):
    z_mean, z_log_var = args
    batch_size = z_mean.get_shape().as_list()[0]
    latent_dim = z_mean.get_shape().as_list()[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class VariationalAutoEncoder(object):
    def __init__(self, x_dim=20, h_dim=10, z_dim=2, num_layers=1,
                 batch_size=100, max_num_iter=100):
        self.is_placeholder = True
        super(VariationalAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.max_num_iter = max_num_iter

        self.h = []
        # Construct Dimensionality Reduction Layers
        self.x = Input(batch_shape=(batch_size, x_dim))
        hop = int((x_dim - h_dim) / num_layers)
        for c_layer in range(num_layers):
            if not c_layer:
                self.h.append(Dense(x_dim - hop, activation='relu')(self.x))
            else:
                self.h.append(Dense(x_dim - hop * (c_layer + 1),
                                    activation='relu')(self.h[c_layer - 1]))

        # Construct Sampling Layer
        self.z_mean = Dense(z_dim)(self.h[c_layer])
        self.z_log_var = Dense(z_dim)(self.h[c_layer])
        self.z = Lambda(sampling, output_shape=(z_dim,))(
            [self.z_mean, self.z_log_var])

        # For generator we both need nodes and the tensors
        self.hd, self.hdf = [], []
        for c_layer in range(num_layers):
            if not c_layer:
                self.hdf.append(Dense(h_dim, activation='relu'))
                self.hd.append(self.hdf[c_layer](self.z))
            elif c_layer == num_layers - 1:
                self.hdf.append(Dense(x_dim - hop, activation='sigmoid'))
                self.hd.append(self.hdf[c_layer](self.hd[c_layer - 1]))
            else:
                self.hdf.append(
                    Dense(h_dim + hop * (c_layer + 1), activation='relu'))
                self.hd.append(self.hdf[c_layer](self.hd[c_layer - 1]))

        self.yf = Dense(x_dim, activation='sigmoid')
        self.y = self.yf(self.hd[c_layer])
        self.vae = Model(self.x, self.y)

        self.encoder = None
        self.generator = None
        self.gen_inp = None
        self.hg = []
        self.yg = None

    def fit_vae(self, train_x, val_x):
        self.vae.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.vae.fit(train_x, train_x, shuffle=True,
                     epochs=self.max_num_iter, batch_size=self.batch_size,
                     validation_data=(val_x, val_x))

        # Set encoder
        self.encoder = Model(self.x, self.z_mean)

        # Set generator
        self.gen_inp = Input(shape=(self.z_dim,))
        for c_layer in range(self.num_layers):
            if not c_layer:
                self.hg.append(self.hdf[c_layer](self.gen_inp))
            else:
                self.hg.append(self.hdf[c_layer](self.hg[c_layer - 1]))

        self.yg = self.yf(self.hg[c_layer])
        self.generator = Model(self.gen_inp, self.yg)

    def vae_loss(self, x, x_dec):
        """ Loss function for variational Autoencoder """
        xnt_loss = self.x_dim * metrics.binary_crossentropy(x, x_dec)
        kl_loss = - 0.5 * K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var),
            axis=-1)
        return xnt_loss + kl_loss


class ConditionalVariationalAutoEncoder(object):
    def __init__(self, x_dim=20, c_dim=2, h_dim=10, z_dim=2, num_layers=1,
                 batch_size=100, max_num_iter=100):
        self.is_placeholder = True
        super(ConditionalVariationalAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.max_num_iter = max_num_iter

        self.h = []
        # Construct Dimensionality Reduction Layers
        self.x = Input(batch_shape=(batch_size, x_dim))
        self.c = Input(batch_shape=(batch_size, c_dim))
        self.xc = concat([self.x, self.c])
        hop = int((x_dim - h_dim) / num_layers)
        for c_layer in range(num_layers):
            if not c_layer:
                self.h.append(Dense(x_dim - hop, activation='relu')(self.xc))
            else:
                self.h.append(Dense(x_dim - hop * (c_layer + 1),
                                    activation='relu')(self.h[c_layer - 1]))

        # Construct Sampling Layer
        self.z_mean = Dense(z_dim)(self.h[c_layer])
        self.z_log_var = Dense(z_dim)(self.h[c_layer])
        self.z = Lambda(sampling, output_shape=(z_dim,))(
            [self.z_mean, self.z_log_var])
        self.zc = concat([self.z, self.c])

        # For generator we both need nodes and the tensors
        self.hd, self.hdf = [], []
        for c_layer in range(num_layers):
            if not c_layer:
                self.hdf.append(Dense(h_dim, activation='relu'))
                self.hd.append(self.hdf[c_layer](self.zc))
            elif c_layer == num_layers - 1:
                self.hdf.append(Dense(x_dim - hop, activation='sigmoid'))
                self.hd.append(self.hdf[c_layer](self.hd[c_layer - 1]))
            else:
                self.hdf.append(
                    Dense(h_dim + hop * (c_layer + 1), activation='relu'))
                self.hd.append(self.hdf[c_layer](self.hd[c_layer - 1]))

        self.yf = Dense(x_dim, activation='sigmoid')
        self.y = self.yf(self.hd[c_layer])
        self.c_vae = Model([self.x, self.c], self.y)

        self.encoder = None
        self.generator = None
        self.gen_inp = None
        self.gen_c_inp = None
        self.hg = []
        self.yg = None

    def fit_c_vae(self, train_x, train_y, val_x, val_y):
        self.c_vae.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.c_vae.fit(train_x, train_y, shuffle=True,
                       epochs=self.max_num_iter, batch_size=self.batch_size,
                       validation_data=(val_x, val_y))

        # Set encoder
        self.encoder = Model([self.x, self.c], self.z_mean)

        # Set generator
        self.gen_inp = Input(shape=(self.z_dim + self.c_dim,))
        for c_layer in range(self.num_layers):
            if not c_layer:
                self.hg.append(self.hdf[c_layer](self.gen_inp))
            else:
                self.hg.append(self.hdf[c_layer](self.hg[c_layer - 1]))

        self.yg = self.yf(self.hg[c_layer])
        self.generator = Model(self.gen_inp, self.yg)

    def vae_loss(self, x, x_dec):
        """ Loss function for variational Autoencoder """
        xnt_loss = self.x_dim * metrics.binary_crossentropy(x, x_dec)
        kl_loss = - 0.5 * K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var),
            axis=-1)
        return xnt_loss + kl_loss


class RecurrentVariationalAutoEncoder(object):
    def __init__(self, x_dim=20, h_dim=10, z_dim=2, hop=10,
                 num_layers=1, batch_size=100, max_num_iter=100):
        self.is_placeholder = True
        super(RecurrentVariationalAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.max_num_iter = max_num_iter

        self.hf = Sequential()
        self.hf.add(TimeDistributed(Dense(self.h_dim, activation='relu'),
                                    input_shape=(self.hop, self.x_dim)))
        for c_layer in range(num_layers):
            if c_layer == num_layers - 1:
                self.hf.add(LSTM(h_dim, return_sequences=False,
                                 input_shape=(hop, h_dim)))
            else:
                self.hf.add(LSTM(h_dim, return_sequences=True,
                                 input_shape=(hop, h_dim)))

        # Construct Sampling Layer
        self.z_inp = Input(batch_shape=(batch_size, h_dim))
        self.z_mean = Dense(z_dim)(self.z_inp)
        self.z_log_var = Dense(z_dim)(self.z_inp)
        self.z = Lambda(sampling, output_shape=(z_dim,))(
            [self.z_mean, self.z_log_var])
        self.hd_inp = Dense(h_dim)(self.z)

        self.zf = Model(self.z_inp, self.hd_inp)

        # For generator we both need nodes and the tensors
        self.hdf = Sequential()
        for c_layer in range(num_layers):
            if c_layer == num_layers - 1:
                self.hdf.add(LSTM(h_dim, return_sequences=False,
                                  input_shape=(hop, x_dim)))
            else:
                self.hdf.add(LSTM(h_dim, return_sequences=True,
                                  input_shape=(hop, x_dim)))

        self.seq_vae = Sequential()
        self.seq_vae.add(self.hf)
        self.seq_vae.add(self.zf)
        self.seq_vae.add(self.hdf)

        self.encoder = None
        self.generator = None
        self.gen_inp = None
        self.hg = []
        self.yg = None

    def fit_r_vae(self, train_x, val_x):
        self.seq_vae.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.seq_vae.fit(train_x, train_x, shuffle=True,
                         epochs=self.max_num_iter, batch_size=self.batch_size,
                         validation_data=(val_x, val_x))

        # Set encoder
        self.encoder = Sequential(self.x, self.z_mean)

        # Set generator
        self.gen_inp = Input(shape=(self.z_dim,))
        for c_layer in range(self.num_layers):
            if not c_layer:
                self.hg.append(self.hdf[c_layer](self.gen_inp))
            else:
                self.hg.append(self.hdf[c_layer](self.hg[c_layer - 1]))

        self.yg = self.yf(self.hg[c_layer])
        self.generator = Sequential(self.gen_inp, self.yg)

    def vae_loss(self, x, x_dec):
        """ Loss function for variational Autoencoder """
        xnt_loss = self.x_dim * metrics.binary_crossentropy(x, x_dec)
        kl_loss = - 0.5 * K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var),
            axis=-1)
        return xnt_loss + kl_loss
