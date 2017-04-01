from chainer import ChainList
from chainer import Chain
from chainer import Variable
from chainer import functions as F
from chainer import links as L
import numpy as np


class ChainerNetwork(ChainList):
    """ Neural Network Chain defined in chainer environment """

    def __init__(self, list_node):
        """ Create the network """
        super().__init__()
        self.length_network = len(list_node)
        for idx in range(self.length_network):
            self.add_link(list_node[idx])

        self.c = [None] * self.length_network
        self.a = [None] * (self.length_network - 1)

    def __call__(self, x, y):
        """ Feed forward of the Network
        Args:
            x(variable): input vector for network
            y(variable): ground truth values for the output
        """

        self.c[0] = self.__getitem__(0)(x)

        for idx in range(self.length_network - 1):
            self.a[idx] = F.relu(self.c[idx])
            self.c[idx + 1] = self.__getitem__(idx + 1)(self.a[idx])

        return F.mean_squared_error(
            F.sigmoid(self.c[self.length_network - 1]), y)


class ChainerAutoEncoder(ChainList):
    """ Auto Encoder Chain defined in chainer environment """

    def __init__(self, list_node):
        """ Create the network
        Args:
            list_note(list): number of hidden layers in the autoencoder
            """
        super().__init__()

        # Assert if number of hidden layers is not odd
        assert len(list_node) % 2 != 1, "Number of hidden layers should be " \
                                        "odd!"
        self.length_network = len(list_node)
        for idx in range(self.length_network):
            self.add_link(list_node[idx])

        self.c = [None] * self.length_network
        self.a = [None] * (self.length_network - 1)

    def __call__(self, x):
        """ Feed forward the Network

        Args:
            x(variable): input vector for network
        Return:
            mean squared error between the estimate and original x
        """

        h = self.encode(x)
        x_hat = self.decode(h)

        return F.mean_squared_error(x_hat, x)

    def encode(self, x):
        """ Using the first half of the network encode the input
        Args:
            x(variable): input
        Return:
            h(variable): code

        """
        idx = 0
        self.c[idx] = self.__getitem__(idx)(x)
        for idx in range((int(self.length_network / 2)) - 1):
            self.a[idx] = F.sigmoid(self.c[idx])
            self.c[idx + 1] = self.__getitem__(idx + 1)(self.a[idx])

        # Check if we are done or need to go further
        if int(self.length_network / 2) - 1 != 0:
            idx += 1

        self.a[idx] = F.sigmoid(self.c[idx])

        h = self.a[idx]

        return h

    def decode(self, h):
        """ Using the second half of the network decode the code
        Args:
            h(variable): code
        Return:
            x_hat(variable): estimate of the input signal
        """

        idx = int(self.length_network / 2)
        self.c[idx] = self.__getitem__(idx)(h)

        for idx in range(int((self.length_network + 1) / 2),
                         int(self.length_network) - 1):
            self.a[idx] = F.sigmoid(self.c[idx])
            self.c[idx + 1] = self.__getitem__(idx + 1)(self.a[idx])

        # Check if we are done or need to go further
        if int(self.length_network / 2) - 1 != 0:
            idx += 1

        x_hat = self.c[idx]
        return x_hat


class chainerRNN(ChainList):
    """ Recurrent Neural Network defined in chainer environment
        Attributes:
            size_inp(int): size of the input vector
            size_state(int): output length of the state
            size_out(int): size of the output vector
            num_layer(int): number of LSTM layers
            drop_ratio(int): forget rate for the LSTM
            activation(ChainerFunction): Activation function
            loss(ChainerFunction): Loss function """

    def __init__(self, size_inp=1, size_state=1, size_out=1, num_layer=1,
                 drop_ratio=0, activation=F.sigmoid,
                 loss=F.softmax_cross_entropy, forget_bias=0):
        """ Initialize the Autoencoder Units """

        super().__init__()
        # Arbitrary links
        self.add_link(L.Linear(size_inp, size_state))
        for i in range(num_layer):
            self.add_link(L.LSTM(size_state, size_state,
                                 forget_bias_init=forget_bias))

        self.add_link(L.Linear(size_state, size_out))

        self.n_layers = num_layer
        self.activation = activation
        self.drop_ratio = drop_ratio
        self.loss = loss

    def __call__(self, x, y, train=False):
        """ Loss function of rnn
            Args:
                x(variable): input variable
                y(variable): grand truth output
                train(bool): dropout flag (make your graph sparse)
            Return:
                loss(variable): loss function we backprop through """

        return self.loss(x=self.feed_forward(x, train=train), t=y)

    def reset_state(self):
        """ Resets all LSTM layers """
        for i in range(1, self.n_layers + 1):
            self[i].reset_state()

    def feed_forward(self, x, train=False):
        """ Feed Forward Neural Network Structure
            Args:
                x(variable): input variable
                train(bool): dropout flag (make your graph sparse)
            Return:
                y(variable): output variable"""
        tmp = self.activation(self[0](x))
        for i in range(1, self.n_layers + 1):
            tmp = self[i](F.dropout(tmp, self.drop_ratio, train))
        y = self[-1](tmp)

        return y

    def predict(self, x, train=False):
        """ Probability distribution function estimator
            Args:
              x(variable): input variable
            Return: probability distribution """
        return F.softmax(self.feed_forward(x, train))


class TrainingPartition(object):
    """ Partitions the given set into training and validation sets
        Partitions the training set into batches
        Attributes:
          data(list[ndarray]): observations
          label(list[ndarray]): labels of observations
          idx(ndarray): shuffled indices dot data
          size_batch(int): batch size
          size_train(int): training set size
          size_val(int): validation set size
          idx_train(ndarray): indices for training set
          idx_val(ndarray): indices for validation set
          idx_epoch(ndarray): indices for epoch
          loss_epoch(float): loss in each epoch
          data_epoch(list[ndarray]): data for epoch
          label_epoch(list[ndarray]): labels for epoch
          idx_batch(ndarray): indices for batch
          data_batch(list[ndarray]): data for batch
          label_batch(list[ndarray]): labels for batch
          eg_val(list[ndarray]): example output data for validation
          """

    def __init__(self, size_obs, size_batch, per_train, data, label):
        """ Initialize the training set with all dependencies
            Args:
              size_obs(int): number of samples
              size_batch(int): batch size
              per_train(float): percentage of training set
              data(list[ndarray]): list of data
              label(list[ndarray]): list of labels """
        self.data = data
        self.label = label

        self.size_batch = size_batch

        self.idx = np.random.permutation(size_obs - 1)
        self.size_train = int(size_obs * per_train)
        self.size_val = int(size_obs * (1 - per_train))

        self.idx_val = self.idx[self.size_val:self.size_val * 2]
        self.data_val = []
        self.label_val = []
        self.loss_val = []

        self.idx_train = list(set(self.idx) - set(self.idx_val))

        self.idx_epoch = []
        self.loss_epoch = []
        self.data_epoch = []
        self.label_epoch = []

        self.idx_batch = 0
        self.data_batch = []
        self.label_batch = []

        self.eg_val = []

    def init_val(self):
        """ Initialize validation """
        self.data_val = [self.data[idx] for idx in self.idx_val]
        self.label_val = [self.label[idx] for idx in self.idx_val]
        self.eg_val = []

    def init_epoch(self):
        """ Initialize the epoch """
        self.idx_epoch = np.random.permutation(self.idx_train)
        self.data_epoch = [self.data[idx] for idx in self.idx_epoch]
        self.label_epoch = [self.label[idx] for idx in self.idx_epoch]

    def update_batch(self, step):
        """ Update batch indices depending on step
            Observe! : This function should be called after init_epoch
            Args:
              step(int): update to batch step """
        self.idx_batch = self.idx_epoch[step: step + self.size_batch]
        self.data_batch = self.data_epoch[step: step + self.size_batch]
        self.label_batch = self.label_epoch[step: step + self.size_batch]

        # Insert zeros as the initial label
        for idx in range(len(self.label_batch)):
            self.label_batch[idx] = [0] + self.label_batch[idx]


class Sampler(object):
    """ Samples dummy data
        Attributes:
            samples(list): samples
            labels(list): label vectors for each observation
            label_seq(list): label_vec that asks where to ask question
            set(set): set of backprop points"""

    def __init__(self, num_obs, fs, len_signal, len_label, offset, var_noise):
        """ Benchmark generation function
            Args:
              num_obs(int): number of observations generated
              fs(int): frequency of the sample
              len_signal(int): length of the samples in [s]
              len_label(int): length of label sequence
              offset(int): response interval of the in samples
              var_noise(float): variance of the noise """
        self.num_samples = fs * len_signal + offset

        self.samples = []
        self.labels = []
        self.label_seq = []
        self.offset = offset

        for i in range(num_obs):
            # Generate Noise signal
            sigma = var_noise
            mu = 0
            sig = np.random.normal(mu, sigma, int(self.num_samples))

            label = np.zeros(int(self.num_samples))
            tar = np.random.randint(0, len_label - 1)

            tmp_sin = np.linspace(-np.pi, np.pi, int(self.num_samples / len_label))
            sig[int(self.num_samples * tar / len_label):int(
                self.num_samples * (tar + 1) / len_label)] += \
                np.sin(tmp_sin)

            label[int(self.num_samples * tar / len_label):int(
                self.num_samples * (tar + 1) / len_label)] = \
                np.ones(int(self.num_samples / len_label))

            self.samples.append(sig)
            self.labels.append(label)
            tmp = np.zeros(len(sig))
            tmp[int(
                self.num_samples * (tar + 1) / len_label) + self.offset] = 1
            self.label_seq.append(tmp)
            self.set = set(list(tar * (i / 2) + offset for i in
                                range(1, int(2 * len_label))))
