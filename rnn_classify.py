import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.functions.connection import n_step_lstm as rnn
from chainer import link
import numpy as np


# TODO: Conditional Variational Autoencoder implementation
# Conditional Variational Autoencoder Implementation
# class CondVAE(chainer.ChainList):
#     """ Conditional Variational Autoencoder Implementation """
#
#     def __init__(self, n_layers=2, decrease_rate_enc=0.5, n_inputs_enc=10,
#                  n_output=2, activation=F.sigmoid):
#         """ Initialization
#              Args:
#                 n_layers(int): number of layers
#                 decrease_rate(float): size_layer_(i)/size_layer_(i+1) ratio
#                 n_inputs(int): input size
#                 n_output(int): output size
#                 activation(chainerF): activation function in MLP """
#
#         super().__init__()
#         for i in range(n_layers - 1):
#             self.add_link(
#                 L.Linear(int(n_inputs_enc * pow(decrease_rate_enc, i)),
#                          int(n_inputs_enc * pow(decrease_rate_enc, i + 1))))
#         self.add_link(
#             L.Linear(int(n_inputs_enc * pow(decrease_rate_enc, i + 1)),
#                      n_output))


# Multilayer Perceptron Implementation
class MLP(chainer.ChainList):
    """ Multi Layer Perceptron implementation"""

    def __init__(self, n_layers=2, decrease_rate=0.5, n_inputs=10,
                 n_output=2, activation=F.sigmoid):
        """ Initialization
             Args:
                n_layers(int): number of layers
                decrease_rate(float): size_layer_(i)/size_layer_(i+1) ratio
                n_inputs(int): input size
                n_output(int): output size
                activation(chainerF): activation function in MLP """

        super().__init__()
        for i in range(n_layers - 1):
            self.add_link(L.Linear(int(n_inputs * pow(decrease_rate, i)),
                                   int(n_inputs * pow(decrease_rate, i + 1))))
        self.add_link(L.Linear(int(n_inputs * pow(decrease_rate, i + 1)),
                               n_output))

        self.n_layers = n_layers
        self.activation = activation

    def __call__(self, x):
        """ Feed forward elements one by one """
        return [self.predict(xk) for xk in x]

    def predict(self, x):
        """ Feed forward neural network """
        tmp = self.activation(self[0](x))
        for i in range(1, self.n_layers - 1):
            tmp = self.activation(self[i](tmp))
        y = self[-1](tmp)

        return y


# Written by F.Quivira
# RNN class with multi layer LSTM
class MultiLayerLSTM(chainer.ChainList):
    def __init__(self, n_units=50, n_layers=2, n_inputs=1, n_classes=2,
                 activation=F.elu, forget_bias=2):
        super().__init__()

        # Arbitrary links
        self.add_link(L.Linear(n_inputs, n_units))
        for i in range(n_layers):
            self.add_link(L.LSTM(n_units, n_units,
                                 forget_bias_init=forget_bias))

        self.add_link(L.Linear(n_units, n_classes))

        self.n_layers = n_layers
        self.activation = activation

    # Override call to compute loss
    def __call__(self, x):
        return self.predict(x)

    # Assume only LSTM layers
    def reset_state(self):
        for i in range(1, self.n_layers + 1):
            self[i].reset_state()

    # Compute output for all time steps
    def predict(self, x):
        return [self._predict_time_step(xk) for xk in x]

    # Compute output given input
    def _predict_time_step(self, x):

        # Compute embedding
        tmp = self.activation(self[0](x))

        for i in range(1, self.n_layers + 1):
            tmp = self[i](tmp)

        # Apply linear layer
        y = self[-1](tmp)
        return y


# RNN class with multi layer LSTM
class MultiLayerNStepLSTM(chainer.Chain):
    def __init__(self, n_units=50, n_layers=2, n_inputs=1, n_classes=1,
                 activation=F.elu):
        super().__init__()

        super(MultiLayerNStepLSTM, self).__init__(
            l_in=L.Linear(n_inputs, n_units),
            l_lstm=NStepLSTM(n_layers, n_units, n_units, 1.0),
            l_out=L.Linear(n_units, n_classes)
        )

        self.activation = activation
        self.n_units = n_units
        self.n_layers = n_layers

        self.h = None
        self.c = None

    # Override call to compute loss
    def __call__(self, x):
        return self.predict(x)

    # Assume only LSTM layers
    def reset_state(self):
        self.h = None
        self.c = None

    # Compute output given input
    def predict(self, x):
        # Compute embedding
        x_hid = [self.activation(self.l_in(xk)) for xk in x]

        if self.h is None:
            batch_size = x_hid[0].shape[0]
            self.h = Variable(
                self.xp.zeros((self.n_layers, batch_size, self.n_units),
                              dtype=self.xp.float32))
            self.c = Variable(
                self.xp.zeros((self.n_layers, batch_size, self.n_units),
                              dtype=self.xp.float32))

        # Apply n step lstm
        self.h, self.c, y_hid = self.l_lstm(self.h, self.c, x_hid, train=False)

        # Apply linear layer
        y = [self.l_out(yk) for yk in y_hid]

        return y


# Mixture density network
class MixtureDensityNetwork(chainer.Chain):
    def __init__(self, predictor, n_inputs=1, n_units=10, n_outputs=1,
                 n_mixtures=3, activation=F.sigmoid):
        self.n_units = n_units
        self.n_outputs = n_outputs
        self.n_mixtures = n_mixtures
        self.activation = activation

        super(MixtureDensityNetwork, self).__init__(
            predictor=predictor,
            coef=L.Linear(n_units, n_mixtures),
            mean=L.Linear(n_units, n_mixtures * n_outputs),
            logvar=L.Linear(n_units, n_mixtures)
        )

    def __call__(self, x, y):
        mean, logvar, coef = self.predict(x)
        neg_log_likelihood = -F.sum(F.log(self.density(y, mean, logvar, coef)))
        return neg_log_likelihood

    def density(self, y, mean, logvar, coef):
        mean, y = F.broadcast(mean, F.reshape(y, (-1, 1, self.n_outputs)))
        return F.sum(
            coef * F.exp(
                -0.5 * F.sum((y - mean) ** 2, axis=2) * F.exp(-logvar)) /
            ((2 * np.pi * F.exp(logvar)) ** (0.5 * self.n_outputs)), axis=1)

    def predict(self, x):
        h = self.activation(self.l1(x))
        coef = F.softmax(self.coef(h))
        mean = F.reshape(self.mean(h), (-1, self.n_mixtures, self.n_outputs))
        logvar = self.logvar(h)
        return (mean, logvar, coef)


# Classifier class
class Classifier(chainer.Chain):
    def __init__(self, predictor, loss_fun=F.softmax_cross_entropy,
                 class_fun=F.softmax):
        super().__init__(predictor=predictor)
        self.loss_fun = loss_fun
        self.class_fun = class_fun

    # Compute loss
    def __call__(self, x, t):

        y = self.predictor(x)

        loss = None

        for yk, tk in zip(y, t):
            if loss is not None:
                loss += self.loss_fun(yk, tk)
            else:
                loss = self.loss_fun(yk, tk)

        return loss

    # Compute normalized posterior
    def classify(self, x):
        y = self.predictor(x)

        return [self.class_fun(yk) for yk in y]

    def reset_state(self):
        self.predictor.reset_state()


class NStepLSTM(chainer.ChainList):
    """Stacked LSTM for sequnces.

    This link is stacked version of LSTM for sequences. It calculates hidden
    and cell states of all layer at end-of-string, and all hidden states of
    the last layer for each time.

    Unlike :func:`chainer.functions.n_step_lstm`, this function automatically
    sort inputs in descending order by length, and transpose the seuqnece.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        use_cudnn (bool): Use cuDNN.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, use_cudnn=True,
                 forget_bias_init=1):
        weights = []
        for i in range(n_layers):
            weight = link.Link()
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = in_size
                else:
                    w_in = out_size
                weight.add_param('w%d' % j, (out_size, w_in))
                weight.add_param('b%d' % j, (out_size,))
                getattr(weight, 'w%d' % j).data[...] = np.random.normal(
                    0, np.sqrt(1. / w_in), (out_size, w_in))

                bias_init = forget_bias_init if j == 1 or j == 5 else 0
                getattr(weight, 'b%d' % j).data[...] = bias_init

            weights.append(weight)

        super(NStepLSTM, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = 0.0
        self.use_cudnn = use_cudnn
        self.out_size = out_size

    def __call__(self, hx, cx, xs, train=False):
        """Calculate all hidden states and cell states.

        Args:
            hx (~chainer.Variable): Initial hidden states.
            cx (~chainer.Variable): Initial cell states.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        """

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, ys = rnn.n_step_lstm(
            self.n_layers, self.dropout, hx, cx, ws, bs, xs,
            train=True, use_cudnn=self.use_cudnn)

        return hy, cy, ys
