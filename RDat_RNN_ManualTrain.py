import chainer
from neuralnet_models import MultiLayerLSTM, Classifier
from sklearn.model_selection import train_test_split
import time
import numpy as np
from chainer import Variable, optimizers
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import winsound

""" Trains a Recurrent Neural Network
    Args:
        Requires time series, each time sample labelled.
        Requires input and training parameters to be adjusted.
    Return:
        An RNN model. """

""" Data should be three dimensional array
    data: [num_samples x length of time signal x number of channels]
    label: [num_samples x 1 x length of time signal]"""

folder_name = '\D12a3a-session2'

# As we are using reshape form Matrix dimensions should be as this
samples_o = sio.loadmat('data.mat')
labels_o = sio.loadmat('label.mat')
raw_samples = samples_o['data_m']
raw_labels = np.concatenate(labels_o['label_m']).astype(np.int32)

samples = [raw_samples[i] for i in range(raw_samples.shape[0])]
labels = [raw_labels[i] for i in range(raw_labels.shape[0])]

# Parameters
ratio_batch = 0.3  # Batch size
valid_size = 0.05  # Validation percentage in training
size_out = 2  # Size of the output layer (If Softmax use 2)
size_inp = samples[0].shape[1]  # Size of the input layer
size_state = 50  # Size of the LSTM
num_layer = 3  # Number of LSTM layers (depth)

grad_clip = 5  # Gradient clipping
back_prop_length = 20  # BPTT rate (accumulate # samples then BP)
forget_bias = 1  # Bias of the forget gate / Required for initialization
max_num_iter = 100  # Maximum number of epochs
valid_epoch = 5  # Perform validation after # of epochs

# Modification starts here
num_samples = samples[0].shape[0]

# Split data
x_train, x_valid, y_train, y_valid = \
    train_test_split(samples, labels, stratify=labels,
                     test_size=valid_size)

x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], size_inp)
x_train = x_train.astype(np.float32)
y_train = np.asarray(y_train)
y_train = y_train.astype(np.int32)
size_batch = int(len(x_train) * ratio_batch)

x_valid = np.asarray(x_valid)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], size_inp)
x_valid = x_valid.astype(np.float32)
y_valid = np.asarray(y_valid)
y_valid = y_valid.astype(np.int32)

# Start Training Network
# Define model and optimizer
model = Classifier(MultiLayerLSTM(n_units=size_state, n_layers=num_layer,
                                  n_classes=size_out, n_inputs=size_inp,
                                  forget_bias=forget_bias))

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

# For all epochs.
indices = [i for i in range(len(x_train))]
for i in range(max_num_iter):

    t0 = time.time()
    model.reset_state()
    random.shuffle(indices)
    x_train = np.asarray([x_train[i] for i in indices])
    y_train = np.asarray([y_train[i] for i in indices])

    # For all mini_batches
    loss_data = 0
    for b_idx in range(int(len(x_train) / size_batch)):
        # Accumulate loss for each data point until right before end
        x_b_train = x_train[b_idx * size_batch: (b_idx + 1) * size_batch, :, :]
        y_b_train = y_train[b_idx * size_batch: (b_idx + 1) * size_batch, :]

        error_c = []
        for k in range(0, num_samples, back_prop_length):
            # Get batches
            trial_length = min(back_prop_length, num_samples - k)
            x = [Variable(x_b_train[:, k + n]) for n in range(trial_length)]
            y = [Variable(y_b_train[:, k + n]) for n in range(trial_length)]

            loss = model(x, y) / trial_length

            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

            loss_data += loss.data

    loss_data /= ((b_idx + 1) * (k / back_prop_length))

    epoch += 1
    print('epoch {}, error {}'.format(epoch, loss_data))
    train_loss.append(loss_data)

    t1 = time.time()
    time_per_epoch += t1 - t0

    # In validation step, copy model and
    # evaluate loss without computational graph
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
plt.savefig('.\FigRNN' + folder_name + '\DSFigTrain.pdf', format='pdf')

# Get best model from validation
model = models[np.argmin(valid_loss)]
model.reset_state()

# Plot probabilities and estimated labels
n_valid = x_valid.shape[0]
x = [Variable(x_valid[:, n]) for n in range(num_samples)]
y = [Variable(y_valid[:, n]) for n in range(num_samples)]

with chainer.no_backprop_mode():
    prob_hat = np.array([pn.data for pn in model.classify(x)])

count = 1
for j, (x, y, p) in enumerate(zip(x_valid, y_valid, prob_hat.swapaxes(0, 1))):
    x = x_valid[j]
    y = y_valid[j]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, 'b')
    ax2.plot(p[:, -1], 'r')
    ax2.plot(y, 'g--')

    ax1.set_xlim(0, num_samples - 1)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel('samples')
    ax1.set_ylabel('data', color='b')

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('prob target', color='r')

    ax1.grid(True)

    ax1.set_title('data and probability signals')
    plt.savefig('.\FigRNN' + folder_name + '\S'+str(count) + 'DSFig.pdf',
                format='pdf')
    count += 1

# Do other stuff while running and let it warn you
Freq = 2500  # Set Frequency To 2500 Hertz
Dur = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq, Dur)
