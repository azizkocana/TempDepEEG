import chainer
from neuralnet_models import MultiLayerLSTM, Classifier, MLP, GenAdvNet
from sklearn.model_selection import train_test_split
import time
import numpy as np
from chainer import Variable, optimizers
from chainer import functions as F
import random
import pickle


class TrainRNN(object):
    """ Trains Recurrent Neural Network given the training set.
        Functions:
            train: trains the RNN
            test: tests RNN """

    def __init__(self, raw_samples, raw_labels, ratio_batch=0.3, test_size=0.1,
                 valid_size=0.1, size_out=2, size_state=50, num_layer=3,
                 bptt=20, max_num_iter=300, valid_epoch=2, print_name='dummy'):
        """ Args:
            raw_samples(ndarray[#_samples x time_samples x channels]): float32
            raw_labels(ndarray[#_samples x time_samples]): int32
            ratio_batch(float): batch size = #_samples * ratio_batch
            test_size(float): Train-Test Split ratio for sample set
            valid_size(float): Validation-Train Split ratio for Train set
            size_out(int): output size for RNN
            size_state(int): state size for LSTM
            num_layer(int): number of LSTM layers
            bptt(int): skip bptt samples then update parameters
            max_num_iter(int): maximum number of iterations
            valid_epoch(int): wait valid_epoch iterations then validate
            print_name(string): name of the model
        """
        self.samples = [raw_samples[i] for i in range(raw_samples.shape[0])]
        self.labels = [raw_labels[i] for i in range(raw_labels.shape[0])]

        self.ratio_batch = ratio_batch
        self.test_size = test_size
        self.valid_size = valid_size  # Validation percentage in training
        self.size_out = size_out  # Size of the output layer (If Softmax use 2)
        self.size_state = size_state  # Size of the LSTM
        self.num_layer = num_layer  # Number of LSTM layers (depth)

        self.grad_clip = 5  # Gradient clipping
        self.back_prop_length = bptt
        self.forget_bias = 1
        self.max_num_iter = max_num_iter  # Maximum number of epochs
        self.valid_epoch = valid_epoch  # Perform validation after # of epochs

        self.size_inp = self.samples[0].shape[1]  # Size of the input layer
        self.num_samples = self.samples[0].shape[0]

        indices = np.arange(len(self.samples))

        # Split data
        self.train_samples, self.test_samples, self.train_labels, \
        self.test_labels, self.train_idx, self.test_idx = train_test_split(
            self.samples, self.labels, indices, stratify=self.labels,
            test_size=self.test_size)

        # Helper variables
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.print_name = print_name

    def train(self, k):
        """ Trains the recurrent Neural Network
            Args:
                k(int): number of cross validations """

        # Split data
        indices_ts = list(np.arange(len(self.train_samples)))
        x_train, x_valid, y_train, y_valid, = train_test_split(
            self.train_samples, self.train_labels,
            stratify=self.train_labels,
            test_size=self.valid_size)

        dat_train = x_train + x_valid
        lab_train = y_train + y_valid

        min_val_e_models, min_val_errors = [], []
        for cv_idx in range(k):

            # Reset the train loss
            self.train_loss = []
            print('cross validation: {}'.format(cv_idx))

            val_indices = indices_ts[
                          int(cv_idx * len(indices_ts) * self.valid_size)
                          :int((cv_idx + 1) * len(
                              indices_ts) * self.valid_size - 1)]
            train_indices = list(set(indices_ts) - set(val_indices))

            x_train = [dat_train[i] for i in train_indices]
            y_train = [lab_train[i] for i in train_indices]
            x_valid = [dat_train[i] for i in val_indices]
            y_valid = [lab_train[i] for i in val_indices]

            x_train = np.asarray(x_train)
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                                      self.size_inp)
            x_train = x_train.astype(np.float32)
            y_train = np.asarray(y_train)
            y_train = y_train.astype(np.int32)
            size_batch = int(len(x_train) * self.ratio_batch)

            x_valid = np.asarray(x_valid)
            x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1],
                                      self.size_inp)
            x_valid = x_valid.astype(np.float32)
            y_valid = np.asarray(y_valid)
            y_valid = y_valid.astype(np.int32)

            # Start Training Network
            # Define model and optimizer
            model = Classifier(
                MultiLayerLSTM(n_units=self.size_state,
                               n_layers=self.num_layer,
                               n_classes=self.size_out, n_inputs=self.size_inp,
                               forget_bias=self.forget_bias))

            optimizer = optimizers.Adam()
            optimizer.setup(model)
            optimizer.add_hook(
                chainer.optimizer.GradientClipping(self.grad_clip))

            epoch = 0
            models = []

            time_per_epoch = 0

            # For all epochs.
            indices = [i for i in range(len(x_train))]
            valid_loss = []
            for i in range(self.max_num_iter):

                t0 = time.time()
                model.reset_state()
                random.shuffle(indices)
                x_train = np.asarray([x_train[i] for i in indices])
                y_train = np.asarray([y_train[i] for i in indices])

                # For all mini_batches
                loss_data = 0
                for b_idx in range(int(len(x_train) / size_batch)):
                    # Accumulate loss for each data point until right before end
                    x_b_train = x_train[
                                b_idx * size_batch: (b_idx + 1) * size_batch,
                                :, :]
                    y_b_train = y_train[
                                b_idx * size_batch: (b_idx + 1) * size_batch,
                                :]

                    for k in range(0, self.num_samples, self.back_prop_length):
                        # Get batches
                        trial_length = min(self.back_prop_length,
                                           self.num_samples - k)
                        x = [Variable(x_b_train[:, k + n]) for n in
                             range(trial_length)]
                        y = [Variable(y_b_train[:, k + n]) for n in
                             range(trial_length)]

                        loss = model(x, y) / trial_length

                        model.cleargrads()
                        loss.backward()
                        loss.unchain_backward()
                        optimizer.update()

                        loss_data += loss.data

                loss_data /= ((b_idx + 1) * (k / self.back_prop_length))

                epoch += 1
                print('epoch {}, error {}'.format(epoch, loss_data))
                self.train_loss.append(loss_data)

                t1 = time.time()
                time_per_epoch += t1 - t0

                # In validation step, copy model and
                # evaluate loss without computational graph
                if (i + 1) % self.valid_epoch == 0:
                    models.append(model.copy())
                    models[-1].reset_state()

                    with chainer.no_backprop_mode():
                        # Get batches
                        x = [Variable(x_valid[:, n]) for n in
                             range(self.num_samples)]
                        y = [Variable(y_valid[:, n]) for n in
                             range(self.num_samples)]

                        loss = models[-1](x, y) / self.num_samples
                        valid_loss.append(loss.data)

            time_per_epoch /= self.max_num_iter
            self.valid_loss.append(valid_loss)
            print('total time per epoch = {} seconds'.format(time_per_epoch))

            # Get best model from validation
            self.min_val_e_model = models[np.argmin(valid_loss)]
            self.min_val_e_model.reset_state()

            # Keep Validation Loss and the minimum error model
            min_val_e_models.append(self.min_val_e_model)
            min_val_errors.append(np.min(valid_loss))

        self.min_val_e_model = min_val_e_models[np.argmin(min_val_errors)]
        self.min_val_e_model.reset_state()

        pickle.dump(self.min_val_e_model,
                    open("models" + str(self.print_name) + ".p", "wb"))

    def test(self):

        x_test = self.test_samples
        y_test = self.test_labels

        x_test = np.asarray(x_test)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],
                                self.size_inp)
        x_test = x_test.astype(np.float32)
        y_test = np.asarray(y_test)
        y_test = y_test.astype(np.int32)

        x = [Variable(x_test[:, n]) for n in range(self.num_samples)]
        y = [Variable(y_test[:, n]) for n in range(self.num_samples)]

        with chainer.no_backprop_mode():
            loss = self.min_val_e_model(x, y) / self.num_samples
            self.test_loss.append(loss.data)


class TrainMLP(object):
    """ Trains a Multi Layer Perceptron
        Functions:
            train: Trains the multi layer perceptron using training set and
                picks the model with least validation error
            test: Tests the least validation error model using the test data
             """

    def __init__(self, raw_samples, raw_labels, ratio_batch=0.3, test_size=0.1,
                 valid_size=0.1, num_layer=3, max_num_iter=300, valid_epoch=2):
        """ Args:
            raw_samples(ndarray[#_samples x time_samples x channels]): float32
            raw_labels(ndarray[#_samples x time_samples]): int32
            ratio_batch(float): batch size = #_samples * ratio_batch
            test_size(float): Train-Test Split ratio for sample set
            valid_size(float): Validation-Train Split ratio for Train set
            size_out(int): output size for RNN
            size_state(int): state size for LSTM
            num_layer(int): number of LSTM layers
            max_num_iter(int): maximum number of iterations
            valid_epoch(int): wait valid_epoch iterations then validate
        """
        self.samples = [np.concatenate(raw_samples[i]) for i in range(
            raw_samples.shape[0])]
        self.labels = [raw_labels[i] for i in range(raw_labels.shape[0])]

        # Parameters
        self.num_layer = num_layer  # Number of layers in MLP
        self.ratio_batch = ratio_batch  # Batch size
        self.valid_size = valid_size  # Validation percentage in training
        self.size_out = self.labels[0].shape[0]
        self.size_inp = self.samples[0].shape[0]  # Size of the input layer

        self.grad_clip = 5  # Gradient clipping
        self.max_num_iter = max_num_iter  # Maximum number of epochs
        self.valid_epoch = valid_epoch  # Perform validation after # of epochs
        self.test_size = test_size

        self.num_samples = self.samples[0].shape[0]
        self.rate_dec = np.power((self.size_out / self.size_inp),
                                 1 / num_layer)

        # Split data
        self.train_samples, self.test_samples, self.train_labels, self.test_labels = train_test_split(
            self.samples, self.labels, stratify=self.labels,
            test_size=self.test_size)

        # Helper variables
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

    def train(self):
        # Split data
        x_train, x_valid, y_train, y_valid = \
            train_test_split(self.train_samples, self.train_labels,
                             stratify=self.train_labels,
                             test_size=self.valid_size)

        x_train = np.asarray(x_train)
        x_train = x_train.astype(np.float32)
        y_train = np.asarray(y_train)
        y_train = y_train.astype(np.int32)
        size_batch = int(len(x_train) * self.ratio_batch)

        x_valid = np.asarray(x_valid)
        x_valid = x_valid.astype(np.float32)
        y_valid = np.asarray(y_valid)
        y_valid = y_valid.astype(np.int32)

        # Start Training Network
        # Define model and optimizer
        model = Classifier(MLP(n_layers=self.num_layer,
                               decrease_rate=self.rate_dec,
                               n_inputs=x_train[0].shape[0],
                               n_output=y_train[0].shape[0], activation=F.elu),
                           loss_fun=F.softmax_cross_entropy,
                           class_fun=F.softmax)

        optimizer = optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.grad_clip))

        epoch = 0
        models = []

        time_per_epoch = 0

        # For all epochs.
        indices = [i for i in range(len(x_train))]
        for i in range(self.max_num_iter):

            t0 = time.time()
            random.shuffle(indices)
            x_train = np.asarray([x_train[i] for i in indices])
            y_train = np.asarray([y_train[i] for i in indices])

            # For all mini_batches
            loss_data = 0
            for b_idx in range(int(len(x_train) / size_batch)):
                # Accumulate loss for each data point until right before end
                x_b_train = x_train[
                            b_idx * size_batch: (b_idx + 1) * size_batch, :]
                y_b_train = y_train[
                            b_idx * size_batch: (b_idx + 1) * size_batch, :]

                x = [Variable(x_b_train)]
                y = [Variable(np.where(y_b_train == 1)[1].astype(np.int32))]

                loss = model(x, y)

                model.cleargrads()
                loss.backward()
                optimizer.update()

                loss_data += loss.data

            # In validation step, copy model and
            # evaluate loss without computational graph
            if (i + 1) % self.valid_epoch == 0:
                models.append(model.copy())

                with chainer.no_backprop_mode():
                    # Get batches
                    x = [Variable(x_valid)]
                    y = [Variable(np.where(y_valid == 1)[1].astype(np.int32))]

                    loss = models[-1](x, y)
                    self.valid_loss.append(loss.data)

            loss_data /= b_idx + 1
            epoch += 1
            print('epoch {}, error {}'.format(epoch, loss_data))
            self.train_loss.append(loss_data)

            t1 = time.time()
            time_per_epoch += t1 - t0

        # Get best model from validation
        self.min_val_model = models[np.argmin(self.valid_loss)]

    def test(self):

        x_test = self.test_samples
        y_test = self.test_labels

        x_test = np.asarray(x_test)
        x_test = x_test.astype(np.float32)
        y_test = np.asarray(y_test)
        y_test = y_test.astype(np.int32)

        x = [Variable(x_test)]
        y = [Variable(np.where(y_test == 1)[1].astype(np.int32))]

        with chainer.no_backprop_mode():
            loss = self.min_val_model(x, y) / self.num_samples
            self.test_loss.append(loss.data)

        y_hat = self.min_val_model.classify(x)
        tmp = 0
        for idx_test in range(y_hat[0].shape[0]):
            tmp += np.equal(y[0][idx_test].data, np.where(
                y_hat[0][idx_test].data == np.max(y_hat[0][idx_test].data)))

        self.acc = tmp[0] / y_hat[0].shape[0]


class TrainOneStepRNN(object):
    """ Trains Recurrent Neural Network given the training set.
        Difference from RNN: The label vector is generated at interest points
        Functions:
            train: trains the RNN
            test: tests RNN """

    def __init__(self, raw_samples, raw_labels, num_bp=10, ratio_batch=0.3,
                 test_size=0.1, valid_size=0.1, size_out=10, size_state=50,
                 num_layer=3, max_num_iter=300, valid_epoch=2):
        """ Args:
            raw_samples(ndarray[#_samples x time_samples x channels]): float32
            raw_labels(ndarray[#_samples x time_samples]): int32
            ratio_batch(float): batch size = #_samples * ratio_batch
            test_size(float): Train-Test Split ratio for sample set
            valid_size(float): Validation-Train Split ratio for Train set
            size_out(int): output size for RNN
            size_state(int): state size for LSTM
            num_layer(int): number of LSTM layers
            num_bp(int): number of backpropagations in an epoch
            max_num_iter(int): maximum number of iterations
            valid_epoch(int): wait valid_epoch iterations then validate
        """
        self.samples = [raw_samples[i] for i in range(raw_samples.shape[0])]
        self.labels = [raw_labels[i] for i in range(raw_labels.shape[0])]

        self.ratio_batch = ratio_batch
        self.test_size = test_size
        self.valid_size = valid_size  # Validation percentage in training
        self.size_out = size_out  # Size of the output layer (If Softmax use 2)
        self.size_state = size_state  # Size of the LSTM
        self.num_layer = num_layer  # Number of LSTM layers (depth)

        self.grad_clip = 5  # Gradient clipping
        self.back_prop_numbers = num_bp
        self.forget_bias = 1
        self.max_num_iter = max_num_iter  # Maximum number of epochs
        self.valid_epoch = valid_epoch  # Perform validation after # of epochs

        self.size_inp = self.samples[0].shape[1]  # Size of the input layer
        self.num_samples = self.samples[0].shape[0]

        # Split data
        self.train_samples, self.test_samples, self.train_labels, self.test_labels = train_test_split(
            self.samples, self.labels, stratify=self.labels,
            test_size=self.test_size)

        # Helper variables
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

    def train(self):
        # Split data
        x_train, x_valid, y_train, y_valid = \
            train_test_split(self.train_samples, self.train_labels,
                             stratify=self.train_labels,
                             test_size=self.valid_size)

        x_train = np.asarray(x_train)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                                  self.size_inp)
        x_train = x_train.astype(np.float32)
        y_train = np.asarray(y_train)
        y_train = y_train.astype(np.int32)
        size_batch = int(len(x_train) * self.ratio_batch)

        x_valid = np.asarray(x_valid)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1],
                                  self.size_inp)
        x_valid = x_valid.astype(np.float32)
        y_valid = np.asarray(y_valid)
        y_valid = y_valid.astype(np.int32)

        # Start Training Network
        # Define model and optimizer
        model = Classifier(
            MultiLayerLSTM(n_units=self.size_state, n_layers=self.num_layer,
                           n_classes=self.size_out, n_inputs=self.size_inp,
                           forget_bias=self.forget_bias))

        optimizer = optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.grad_clip))

        epoch = 0
        models = []

        time_per_epoch = 0

        # For all epochs.
        indices = [i for i in range(len(x_train))]
        for i in range(self.max_num_iter):

            t0 = time.time()
            model.reset_state()
            random.shuffle(indices)
            x_train = np.asarray([x_train[i] for i in indices])
            y_train = np.asarray([y_train[i] for i in indices])

            # For all mini_batches
            loss_data = 0
            for b_idx in range(int(len(x_train) / size_batch)):
                # Accumulate loss for each data point until right before end
                x_b_train = x_train[
                            b_idx * size_batch: (b_idx + 1) * size_batch, :, :]
                y_b_train = y_train[
                            b_idx * size_batch: (b_idx + 1) * size_batch, :]

                # Assuming origin is located at 0
                trial_length = x_b_train.shape[1] - 1
                x = [Variable(x_b_train[:, n]) for n in range(trial_length)]
                y_tmp = np.where(y_b_train == 1)[1]
                y = [Variable(y_tmp.astype(np.int32))] * trial_length

                y_hat = model.classify(x)

                loss = F.softmax_cross_entropy(y_hat[-1], y[-1])
                loss /= x[0].shape[0]

                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()

                loss_data += loss.data

            epoch += 1
            print('epoch {}, error {}'.format(epoch, loss_data))
            self.train_loss.append(loss_data)

            t1 = time.time()
            time_per_epoch += t1 - t0

            # In validation step, copy model and
            # evaluate loss without computational graph
            if (i + 1) % self.valid_epoch == 0:
                models.append(model.copy())
                models[-1].reset_state()

                x = [Variable(x_valid[:, n]) for n in range(trial_length)]
                y_tmp = np.where(y_valid == 1)[1]
                y = [Variable(y_tmp.astype(np.int32))] * trial_length
                with chainer.no_backprop_mode():
                    y_hat = models[-1].classify(x)

                    loss = F.softmax_cross_entropy(y_hat[-1], y[-1])
                    loss /= x[0].shape[0]
                    self.valid_loss.append(loss.data)

        time_per_epoch /= self.max_num_iter
        print('total time per epoch = {} seconds'.format(time_per_epoch))

        # Get best model from validation
        self.min_val_e_model = models[np.argmin(self.valid_loss)]
        self.min_val_e_model.reset_state()

    def test(self):

        x_test = self.test_samples
        y_test = self.test_labels

        x_test = np.asarray(x_test)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],
                                self.size_inp)
        x_test = x_test.astype(np.float32)
        y_test = np.asarray(y_test)
        y_test = y_test.astype(np.int32)

        with chainer.no_backprop_mode():
            trial_length = x_test.shape[1] - 1
            x = [Variable(x_test[:, n]) for n in
                 range(trial_length - 1)]
            y_tmp = np.where(y_test == 1)[1]
            y = [Variable(y_tmp.astype(np.int32))] * (trial_length - 1)

            y_hat = self.min_val_e_model.classify(x)
            loss = F.softmax_cross_entropy(y_hat[-1], y[-1])
            loss /= x[0].shape[0]
            self.test_loss.append(loss.data)

        y_hat = self.min_val_e_model.classify(x)
        tmp = 0
        for idx_test in range(y_hat[0].shape[0]):
            tmp += np.equal(y[-1][idx_test].data, np.where(
                y_hat[-1][idx_test].data == np.max(y_hat[-1][idx_test].data)))

        self.acc = tmp[0] / y_hat[0].shape[0]


class TrainGAN(object):
    """ Trains Generative Adversarial Network """

    def __init__(self, raw_samples, mean_gen, cov_gen, num_class=1,
                 size_batch=0.3, test_size=0.1, valid_size=0.1, num_layer=3,
                 max_num_iter=300, valid_epoch=2):
        """ Args:
            raw_samples(ndarray[#_samples x time_samples x channels]): float32
            size_batch(float): batch size = #_samples * ratio_batch
            test_size(float): Train-Test Split ratio for sample set
            valid_size(float): Validation-Train Split ratio for Train set
            mean_gen(nd.array): mean of the generator variable
            cov_gen(nd.array): covariance of the generator variable
            num_class(int): number of classes generated using z

        """

        self.samples = raw_samples

        # Parameters
        self.num_layer = num_layer  # Number of layers in MLP
        self.size_batch = size_batch  # Batch size
        self.valid_size = valid_size  # Validation percentage in training\

        self.size_y = num_class + 1  # Number of classes
        self.size_x = self.samples[0].shape[0]  # Size of the input layer
        self.size_z = mean_gen.shape[0]  # Generator random variable size
        self.mean_z = mean_gen
        self.cov_z = cov_gen

        self.dec_gen = np.power((self.size_x / self.size_z),
                                1 / num_layer)
        self.dec_dis = np.power((self.size_y / self.size_x),
                                1 / num_layer)

        self.grad_clip = 5  # Gradient clipping
        self.max_num_iter = max_num_iter  # Maximum number of epochs
        self.valid_epoch = valid_epoch  # Perform validation after # of epochs
        self.test_size = test_size

        self.num_samples = self.samples[0].shape[0]

        # Helper variables
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.discriminator = MLP(n_layers=self.num_layer,
                                 decrease_rate=self.dec_dis,
                                 n_inputs=self.size_x,
                                 n_output=self.size_y,
                                 activation=F.sigmoid)
        self.generator = MLP(n_layers=self.num_layer,
                             decrease_rate=self.dec_gen, n_inputs=self.size_z,
                             n_output=self.size_x, activation=F.sigmoid)

    def train(self, train_disc):

        chainer.initializers.Identity()

        # Have two different optimizers
        opt_g = optimizers.Adam()
        opt_d = optimizers.Adam()
        opt_g.setup(self.generator)
        opt_d.setup(self.discriminator)
        opt_g.add_hook(chainer.optimizer.GradientClipping(self.grad_clip))
        opt_d.add_hook(chainer.optimizer.GradientClipping(self.grad_clip))

        # For all epochs.
        for i in range(self.max_num_iter):

            random.shuffle(self.samples, random.random)

            acc_loss_d = 0
            for k in range(train_disc):
                for b_idx in range(int(len(self.samples) / self.size_batch)):
                    # Sample from the random distribution
                    z = np.random.multivariate_normal(self.mean_z, self.cov_z,
                                                      self.size_batch)
                    z = [z[c] for c in range(z.shape[0])]
                    z = np.asarray(z)
                    z = z.astype(np.float32)
                    z = [Variable(z)]

                    # Generate fake samples
                    gz = self.generator(z)

                    # Sample from the data set
                    x = self.samples[
                        b_idx * self.size_batch:(b_idx + 1) * self.size_batch]
                    x = np.asarray(x)
                    x = x.astype(np.float32)
                    x = [Variable(x)]

                    c = np.concatenate(np.array([gz[0].data, x[0].data]),
                                       axis=0)
                    c = c.astype(np.float32)
                    c = [Variable(c)]

                    tmp = Variable(
                        np.array(
                            [1 for i in range(x[0].shape[0])] +
                            [0 for i in range(gz[0].shape[0])]).astype(
                            np.int32))
                    loss_d = F.softmax_cross_entropy(self.discriminator(c)[0],
                                                     tmp)

                    self.discriminator.cleargrads()
                    loss_d.backward()
                    opt_d.update()

                    acc_loss_d += loss_d.data

            acc_loss_d /= train_disc * len(
                self.samples) / self.size_batch

            z = np.random.multivariate_normal(self.mean_z, self.cov_z,
                                              self.size_batch)
            z = [z[c] for c in range(z.shape[0])]
            z = np.asarray(z)
            z = z.astype(np.float32)
            z = [Variable(z)]

            dz = self.discriminator(self.generator(z))
            tmp = Variable(
                np.array([1 for i in range(dz[0].shape[0])]).astype(
                    np.int32))
            loss_g = F.softmax_cross_entropy(dz[0], tmp)

            self.generator.cleargrads()
            loss_g.backward()
            opt_g.update()

            print('G:{}, D:{} - Epoch:{}'.format(loss_g.data, acc_loss_d,
                                                 i))
