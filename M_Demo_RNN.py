import chainer
from chainer_network import Sampler
from rnn_classify import MultiLayerLSTM, Classifier
from sklearn.model_selection import train_test_split
import time
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import Variable, optimizers
import matplotlib.pyplot as plt

# Specify sampling frequency
fs = 80
# Generate Dummy Samples
num_inter = 10
max_num_iter = 50
num_obs = 100
size_batch = 90
sig_seconds = 5
offset = 20
valid_size = 0.1
var_noise = 0
valid_epoch = 10

# RNN parameters
val_threshold = 1
back_prop_length = 20
num_layer = 2
size_state = 50
size_inp = 1
size_out = 2
forget_bias = 2
grad_clip = 5

samples = Sampler(int(num_obs), fs, sig_seconds, num_inter, offset,
                  np.sqrt(var_noise + 0.00001))
num_samples = samples.num_samples

# Split data
x_train, x_valid, y_train, y_valid = \
    train_test_split(samples.samples, samples.labels, stratify=samples.labels,
                     test_size=valid_size)

x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_train = x_train.astype(np.float32)
y_train = np.asarray(y_train)
y_train = y_train.astype(np.int32)

x_valid = np.asarray(x_valid)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
x_valid = x_valid.astype(np.float32)
y_valid = np.asarray(y_valid)
y_valid = y_valid.astype(np.int32)

# Start Training Network
# Define model and optimizer
model = Classifier(MultiLayerLSTM(n_units=size_state, n_layers=num_layer,
                                  n_classes=size_out))

optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Helper variables
accum_loss = 0
loss = 0
train_loss = []
valid_loss = []

epoch = 0
loss_data = 0
models = []

time_per_epoch = 0

# For all epochs. No minibatch in this case
for i in range(max_num_iter):

    t0 = time.time()

    model.reset_state()

    # Accumulate loss for each data point until right before end
    for k in range(0, num_samples, back_prop_length):
        # Get batches
        x = [Variable(x_train[:, k + n]) for
             n in range(back_prop_length)]
        y = [Variable(y_train[:, k + n]) for n in range(back_prop_length)]

        loss = model(x, y) / back_prop_length

        model.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

    loss_data = loss.data
    optimizer.update()

    epoch += 1
    print('epoch {}, error {}'.format(epoch, loss_data))
    train_loss.append(loss_data)

    t1 = time.time()
    time_per_epoch += t1 - t0

    # In validation step, copy model and evaluate loss without computational
    # graph
    if (i + 1) % valid_epoch == 0:
        models.append(model.copy())
        models[-1].reset_state()

        with chainer.no_backprop_mode():
            # Get batches
            x = [Variable(x_valid[:, n]) for n in range(num_samples)]
            y = [Variable(y_valid[:, n]) for n in range(num_samples)]

            loss = models[-1](x, y) / num_samples
            valid_loss.append(loss.data)

time_per_epoch = time_per_epoch / max_num_iter
print('total time per epoch = {} seconds'.format(time_per_epoch))

# Plot error over iterations
plt.figure()
plt.plot(np.arange(1, max_num_iter + 1), train_loss, label='train loss')
plt.plot(np.arange(valid_epoch, max_num_iter + 1, step=valid_epoch),
         valid_loss, label='valid loss')
plt.title('softmax cross entropy')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(loc='upper right', shadow=True)
plt.grid(True)
plt.show()

# Get best model from validation
model = models[np.argmin(valid_loss)]
model.reset_state()

# Plot probabilities and estimated labels
n_valid = x_valid.shape[0]
x = [Variable(x_valid[:, n]) for n in range(num_samples)]
y = [Variable(y_valid[:, n]) for n in range(num_samples)]

with chainer.no_backprop_mode():
    prob_hat = np.array([pn.data for pn in model.classify(x)])

for j, (x, y, p) in enumerate(zip(x_valid, y_valid, prob_hat.swapaxes(0, 1))):
    x = x_valid[j]
    y = y_valid[j]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, 'b')
    ax2.plot(p[:, -1], 'r')
    ax2.plot(y, 'g--')

    ax1.set_xlim(0, num_samples - 1)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('samples')
    ax1.set_ylabel('data', color='b')

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('prob target', color='r')

    ax1.grid(True)

    ax1.set_title('data and probability signals')
    plt.show()
