import chainer
from neuralnet_models import MultiLayerLSTM, Classifier
from sklearn.model_selection import train_test_split
import time
import numpy as np
from chainer import Variable, optimizers
import matplotlib.pyplot as plt
import scipy.io as sio
from TrainingSet import TrainRNN
import os
import pickle
import random
import winsound

""" Used for the conference paper for MLSP2017 """

""" Trains a Recurrent Neural Network
    Args:
        Requires time series, each time sample labelled.
        Requires input and training parameters to be adjusted.
    Return:
        An RNN model. """

""" Data should be three dimensional ar ray
    data: [num_samples x length of time signal x number of channels]
    label: [num_samples x 1 x length of time signal]"""

folder_dat = \
    '.\dat\modified_CSL_RSVPKeyboard_di830609_IRB130107_ERPCalibration_2016-07-22-T-16-48'
# If \a kinda happens in print name add M for model in front
print_name = '\di830609_IRB130107_2016-07-22-T-16-48_v2'

# As we are using reshape form Matrix dimensions should be as this
samples_o = sio.loadmat(folder_dat + '\data.mat')
labels_o = sio.loadmat(folder_dat + '\label.mat')
start_idx = sio.loadmat(folder_dat + '\stime.mat')
trial_label_o = sio.loadmat(folder_dat + '\T_lab.mat')
raw_samples = samples_o['data_m']
raw_labels = np.concatenate(labels_o['label_m']).astype(np.int32)
trial_time_idx = start_idx['start_m']
trial_label = trial_label_o['trial_lab_m']
fs = 256

trial_time = np.floor(np.mean(trial_time_idx, axis=0))[0]
prob_letter = np.ones(np.max(trial_label)) * np.power(10, 6)

# Parameters
cross_val_num = 10
num_layer = 4  # Number of LSTM layers (depth)
size_layer = 35
size_out = 2
max_num_iter = 120
test_size = .1

RNN = TrainRNN(raw_samples, raw_labels, ratio_batch=0.3, test_size=test_size,
               valid_size=0.1, size_out=size_out, size_state=size_layer,
               num_layer=num_layer, bptt=20, max_num_iter=max_num_iter,
               valid_epoch=2, print_name=print_name)

if os.path.exists('models' + print_name + '.p'):
    model = pickle.load(open("models" + print_name + ".p", "rb"))
else:

    RNN.train(k=cross_val_num)

    plt.figure()
    plt.plot(np.arange(1, RNN.max_num_iter + 1),
             RNN.train_loss,
             label='train loss')
    for idx_val_plot in range(5):
        plt.plot(
            np.arange(RNN.valid_epoch, RNN.max_num_iter + 1,
                      step=RNN.valid_epoch, ), RNN.valid_loss[idx_val_plot],
            label='valid loss_' + str(idx_val_plot))
    plt.xlabel('iteration[number]')
    plt.ylabel('loss[softmax cross entropy]')
    plt.title('layers: {}x{}'.format(num_layer, size_layer))
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True)
    plt.savefig('.\FigDummy' + print_name + '_Train.pdf', format='pdf')
    model = RNN.min_val_e_model

x_test = RNN.test_samples
y_test = RNN.test_labels

x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],
                        RNN.size_inp)
x_test = x_test.astype(np.float32)
y_test = np.asarray(y_test)
y_test = y_test.astype(np.int32)

x = [Variable(x_test[:, n]) for n in range(RNN.num_samples)]
y = [Variable(y_test[:, n]) for n in range(RNN.num_samples)]

with chainer.no_backprop_mode():
    prob_hat = np.array([pn.data for pn in model.classify(x)])

count = 1
P = []
plt.style.use('grayscale')
for j, (x, y, p) in enumerate(
        zip(RNN.test_samples, RNN.test_labels, prob_hat.swapaxes(0, 1))):
    x = RNN.test_samples[j]
    y = RNN.test_labels[j]

    tmp = np.array([i for i in range(RNN.num_samples)]) / 256

    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    ax2 = ax1.twinx()
    ax1.plot(tmp, x, linewidth=2)
    ax2.plot(tmp, p[:, -1], '.-', label='estimate', linewidth=2)
    ax2.plot(tmp, y, '--', label='true', linewidth=2)

    ax1.set_xlim(0, (RNN.num_samples) / 256)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel('time [s]', fontsize=18)
    ax1.set_ylabel('data [uV]', fontsize=18)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('probability', fontsize=18)
    ax2.legend(loc='upper right', shadow=True, fontsize=15)

    ax1.grid(True)

    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_right()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    # plt.savefig('.\FigRNN' + print_name + '\S' + str(count) + 'DSFig.pdf',
    #           format='pdf', bbox_inches='tight', pad_inches=0.0, dpi=1200)
    plt.savefig('.\FigRNN' + print_name + '\S' + str(count) + 'DSFig.jpg',
                format='jpg', bbox_inches='tight', pad_inches=0.0, dpi=1200)

    p = np.power(p, 1 / p.shape[0])
    p.astype(np.float64)

    tmp = [np.prod(p[:, 0])]
    for idx in range(len(trial_time)):
        tmp.append(
            np.prod(p[int(trial_time[idx]):int(trial_time[idx] + fs / 2), -1])
            / (np.prod(p[int(trial_time[idx]):int(trial_time[idx] + fs / 2),
                       0]) + np.power(.1, 6)) * np.prod(p[:, 0]))

    P.append(np.array(tmp))
    # p_this = np.array(tmp)
    #
    # letter_labels = trial_label[RNN.test_idx[count - 1]]
    # for l in range(np.max(trial_label)):
    #     if l in set(letter_labels[0]):
    #         prob_letter[l] *= p_this[
    #             int(1 + np.where(letter_labels[0] == l)[0][0])]
    #     else:
    #         prob_letter[l] *= (p_this[0] / (np.max(trial_label)
    #                                         - letter_labels.shape[1]))
    #
    # prob_letter *= np.pow(10, 6) / np.max(prob_letter)
    #
    # a = plt.figure()
    # plt.stem(np.array(prob_letter) / np.sum(prob_letter))
    # plt.pause(.5)

    count += 1

# For AUC Calc in Matlab
sio.savemat(folder_dat + '\P.mat', {'P': np.asarray(P)})
sio.savemat(folder_dat + '\pr.mat', {'pr': RNN.test_idx})
# Do other stuff while running and let it warn you
Freq = 2500  # Set Frequency To 2500 Hertz
Dur = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq, Dur)
