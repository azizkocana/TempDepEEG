from chainer_network import chainerRNN, TrainingPartition, Sampler
import time
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import Variable, optimizers
import matplotlib.pyplot as plt
import random

dat_set = [13]
# Read Data
data_hold, dat_labels = [], []
for i in list(dat_set):
    infile = 'dataTrials\data_' + str(i) + '.npy'
    data_hold.append(np.load(infile))
    infile = 'dataTrials\Dtar_' + str(i) + '.npy'
    dat_labels.append(np.load(infile))

# Construct labels
label_hold = []
for i in range(len(data_hold[0])):
    if dat_labels[0][i] == 1:
        label_hold.append(np.ones(128))
    else:
        label_hold.append(np.zeros(128))

# Balance Data
print('Balancing Data...')
idx_ones = np.where(dat_labels[0] == 1)[0]
idx_others = list(set(range(data_hold[0].shape[0])) - set(idx_ones))

targets_n, samples_n = [], []
for idx in idx_ones:
    tmp = []
    for i in range(16):
        tmp.append(data_hold[0][idx][128 * i: 128 * (i + 1)])

    samples_n.append(np.transpose(np.asarray(tmp)))
    targets_n.append(label_hold[idx])

idx_oth = random.sample(idx_others, len(idx_ones))
for idx in idx_oth:
    tmp = []
    for i in range(16):
        tmp.append(data_hold[0][idx][128 * i: 128 * (i + 1)])

    samples_n.append(np.transpose(np.asarray(tmp)))
    targets_n.append(label_hold[idx])

# RNN parameters
max_num_iter = 100
sig_len = len(samples_n[0])
fs = 256
val_threshold = 1
back_prop_rate = 10
num_layer = 3
size_state = 60
size_inp = 16
size_out = 2
size_batch = 10
forget_bias = 10
grad_clip = 5

# Start Training Network
print('Training Network...')
for train_idx in range(5):

    # Define RNN
    train = TrainingPartition(len(samples_n), size_batch, 0.9,
                              samples_n, targets_n)
    rnn = chainerRNN(size_inp=size_inp, size_state=size_state,
                     size_out=size_out, num_layer=num_layer, drop_ratio=0,
                     forget_bias=forget_bias)
    optimizer = optimizers.Adam()
    optimizer.setup(rnn)

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
            asd, asd2 = [], []
            for idx_time in range(sig_len):

                data_time, label_time = [], []
                for b in range(size_batch):
                    data_time.append(
                        [np.asarray(train.data_batch)[b, idx_time]])
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

                asd.append(data_time[0])
                asd2.append(label_time[0])

            asd = np.concatenate(np.asarray(asd), 0)
            plt.plot(np.squeeze(np.asarray(asd)))
            plt.suptitle('Target Flag=' + str(asd2[0]), fontsize=20)
            plt.ylabel('Magnitude[mV?]', fontsize=15)
            plt.xlabel('Samples', fontsize=15)
            #plt.plot(np.asarray(asd2))
            plt.show()

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

                if epoch == max_num_iter - 1:
                    train.eg_val.append(rnn.predict(data_time_val)[0][1].data)

            train.loss_val.append(error_val)
            # print("Val - Loss: {}".format(error_val))

        print("Epoch: {}, Loss: {}".format(epoch + 1, avg_loss))
        val_count += 1

    plt.clf()
    plt.plot(np.asarray(train.loss_epoch), linewidth=2,
             label='epoch-' + str(train_idx))
    plt.plot(np.asarray(train.loss_val), linewidth=2,
             label='validation-' + str(train_idx))
    plt.xlabel('num_epochs', fontsize=15)
    plt.ylabel('Cost', fontsize=15)
    plt.suptitle('Training Errors', fontsize=20)
    plt.legend(loc=1, prop={'size': 12}, fontsize=15)
    # plt.show()
    plt.savefig(
        'TrialTrain' + str(train_idx + 1) + 'L' + str(num_layer) + '_' +
        str(size_state) + '.pdf', format='pdf')

    plt.clf()
    plt.plot(np.asarray(train.label_val[0]), linewidth=2,
             label='original-' + str(train_idx))
    plt.plot(np.asarray(train.eg_val), linewidth=2,
             label='estimate-' + str(train_idx))
    plt.xlabel('num_epochs', fontsize=15)
    plt.ylabel('Value', fontsize=15)
    plt.suptitle('Label Estimate', fontsize=20)
    plt.legend(loc=1, prop={'size': 12}, fontsize=15)
    # plt.show()
    plt.savefig('TrialDat' + str(train_idx + 1) + 'L' + str(num_layer) + '_' +
                str(size_state) + '.pdf', format='pdf')
