from chainer_network import chainerRNN, TrainingPartition, Sampler
import time
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import Variable, optimizers
import matplotlib.pyplot as plt


# Define some functions which are required for trigger partitioning
def request_output(x, fs):
    """ Creates a vector
        Args:
            x(ndarray): input signal as a column vector
            fs(int): sampling frequency of the system
        Return:
            y(ndarray): flag vector """

    y = [0] * x.shape[0]
    for i in range(x.shape[0] - 1):
        if x[i + 1] < x[i]:
            y[int(i + fs * 0.5)] = 1

    return np.asarray(y)

# def loss_func():

# Initialize Params
dat_set = [1]

# Sampling Frequency
fs = 256
cardinality_seq = 16
size_channel = 16
offset = 128

# RNN parameters
val_threshold = 5
max_num_iter = 120
forget_bias = 5
size_inp = size_channel
size_state = 50  # Experience with EMG - probably larger than inputs
size_out = 2
num_layer = 3
size_batch = 3
length_seq = 1400  # TODO: Calculate it
down_smp_rate = 2
back_prop_rate = 100
sig_len = int(length_seq / down_smp_rate)

# Read Data
data_hold, label_hold, trigger_hold = [], [], []
for i in list(dat_set):
    infile = 'data\data_' + str(i) + '.npy'
    data_hold.append(np.load(infile))
    infile = 'data\ger_' + str(i) + '.npy'
    trigger_hold.append(np.load(infile))

# Find the indexes of output request and start points of sequences
flag_out = np.where(request_output(trigger_hold[0], fs) == 1)[0]
start_idx, out_idx, tar_hold = [], [], []
for i in range(flag_out.shape[0]):
    if i % (cardinality_seq + 1) == 0:
        start_idx.append(flag_out[i] - fs *
                         0.5)
        label_tar = (trigger_hold[0][int(flag_out[i] - fs * 0.5)]) - offset
    else:
        out_idx.append(flag_out[i])
        tar_hold.append(
            int(label_tar == trigger_hold[0][int(flag_out[i] - fs * 0.5)]))

# Divide data into sequences
samples = data_hold[0]
label_all, data_all = [], []
for idx in range(len(start_idx)):
    label_all.append(tar_hold[int(idx * cardinality_seq):int(
        (idx + 1) * cardinality_seq):1])

    data_all.append(samples[int(start_idx[int(idx)]):int(
        start_idx[int(idx)] + length_seq):1])

# Calculate interval between flashes and label out stamps
int_smp = min(np.array(out_idx[1:-1:1]) - np.array(out_idx[0:-2:1]))
# Calculate flash indices for a sequence
flash_set = list(
    length_seq - int_smp * np.array(list(range(cardinality_seq))) - fs * 0.5)

# Form the label vector using the information
label_vec_all = []
for idx in range(len(start_idx)):
    tmp = np.zeros(data_all[0].shape[0])
    tmp2 = flash_set[np.where(np.asarray(label_all[idx]) == 1)[0][0]]
    tmp[int(tmp2):(int(tmp2 + 0.5 * fs))] = np.ones(int(fs * 0.5))

    label_vec_all.append(tmp)

# Downsample Signal
for idx in range(len(start_idx)):
    label_vec_all[idx] = label_vec_all[idx][:: down_smp_rate]
    data_all[idx] = data_all[idx][:: down_smp_rate]

# Start Training Here - Actual EEG
train = TrainingPartition(len(start_idx), size_batch, 0.9,
                          data_all, label_vec_all)
rnn = chainerRNN(size_inp=size_inp, size_state=size_state,
                 size_out=size_out, num_layer=num_layer, drop_ratio=0,
                 forget_bias=forget_bias)
optimizer = optimizers.Adam()

optimizer.setup(rnn)


# Generate DUMMY samples
var_noise = 0.5
num_obs = 20
fs = 80
sig_seconds = 10
offset = 10
len_label = 10
samples = Sampler(int(num_obs), fs, sig_seconds, len_label, offset, var_noise)
sig_len = samples.num_samples
# RNN parameters
val_threshold = 5
back_prop_rate = int(fs /2)
num_layer = 2
size_state = 5
size_inp = 1
size_out = 2
size_batch = 4
max_num_iter = 200
forget_bias = 2
grad_clip = 5


# Dummy Data Generation
train = TrainingPartition(num_obs, size_batch, 0.9,
                          samples.samples, samples.labels)
rnn = chainerRNN(size_inp=size_inp, size_state=size_state,
                 size_out=size_out, num_layer=num_layer, drop_ratio=0,
                 forget_bias=forget_bias)
optimizer = optimizers.Adam()

optimizer.setup(rnn)

# Start Training Network
print('Training Network...')
val_count = 0
train.init_val()
for epoch in range(int(max_num_iter) + 1):
    # Initialize epoch and reset LSTM states
    train.init_epoch()
    rnn.reset_state()
    loss_total = 0

    for counter_batch in range(int(train.size_train / size_batch)):
        # Update batch data and labels
        train.update_batch(counter_batch)

        # Initialize time label
        cnt_time = 0

        accum_loss = 0
        #  asd, asd2 = [], []
        for idx_time in range(sig_len):

            data_time, label_time = [], []
            for b in range(size_batch):
                data_time.append([np.asarray(train.data_batch)[b, idx_time]])
                label_time.append(train.label_batch[b][idx_time])

            label_time = np.asarray(label_time).astype(np.int32)
            data_time = np.asarray(data_time).astype(np.float32)

            out = rnn(Variable(data_time), Variable(label_time))
            accum_loss += out

            if idx_time % back_prop_rate == 0:
                rnn.cleargrads()
                accum_loss.backward()
                accum_loss.unchain_backward()
                optimizer.update()
                loss_total += accum_loss.data
                accum_loss = 0

                # asd.append(data_time[1])
                # asd2.append(label_time[1])

                # asd = np.concatenate(np.asarray(asd), 0)
                # plt.plot(np.squeeze(np.asarray(asd[:, 0])))
                # plt.plot(np.asarray(asd2))
                # plt.show()

    avg_loss = loss_total / int(train.size_train / size_batch)
    train.loss_epoch.append(avg_loss)

    # Validation
    if val_count % val_threshold == 0:
        rnn.reset_state()

        error_val = 0
        for idx_time in range(sig_len):

            data_time, label_time = [], []
            for b in range(train.size_val):
                data_time.append([np.asarray(train.data_val)[b, idx_time]])
                label_time.append(train.label_val[b][idx_time])

            label_time_val = np.asarray(label_time).astype(np.int32)
            data_time_val = np.asarray(data_time).astype(np.float32)

            # label = rnn.predict(Variable(data_time_val))
            # label = (label[:, 1].data > 0.5).astype(np.float32)
            # error_val += (F.mean_squared_error(label, label_time_val)).data \
            #             / 2

            error_val += \
                rnn(Variable(data_time_val), Variable(label_time_val)).data

        train.loss_val.append(error_val)
        print("Val - Loss: {}".format(error_val))

    print("Epoch: {}, Loss: {}".format(epoch + 1, avg_loss))
    val_count += 1
