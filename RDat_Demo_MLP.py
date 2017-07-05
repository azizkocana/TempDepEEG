import chainer
from neuralnet_models import MultiLayerLSTM, Classifier
from sklearn.model_selection import train_test_split
import time
import numpy as np
from chainer import Variable, optimizers
import matplotlib.pyplot as plt
import scipy.io as sio
from TrainingSet import TrainRNN, TrainMLP
import os
import pickle
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

folder_dat = \
        '.\dat\modified_CSL_RSVPKeyboard_di830609_IRB130107_ERPCalibration_2016-07-22-T-16-48'
print_name = '\D12a3a-session2-MLP5'

# As we are using reshape form Matrix dimensions should be as this
samples_o = sio.loadmat(folder_dat + '\data.mat')
labels_o = sio.loadmat(folder_dat + '\Tar.mat')
start_idx = sio.loadmat(folder_dat + '\stime.mat')
trial_label_o = sio.loadmat(folder_dat + '\T_lab.mat')
raw_samples = samples_o['data_m']
raw_labels = np.concatenate(labels_o['target_m']).astype(np.int32)
trial_time_idx = start_idx['start_m']
trial_label = trial_label_o['trial_lab_m']


# Parameters
num_layer = 20  # Number of LSTM layers (depth)
size_layer = 50
size_out = 2
max_num_iter = 200
test_size = .1

MLP = TrainMLP(raw_samples, raw_labels, ratio_batch=0.2, test_size=test_size,
               valid_size=0.1, num_layer=num_layer, max_num_iter=max_num_iter,
               valid_epoch=2)

if os.path.exists('models' + print_name + '.p'):
    MLP.min_val_model = pickle.load(open("models" + print_name + ".p", "rb"))
else:
    MLP.train()
    pickle.dump(MLP.min_val_model, open("models" + print_name + ".p", "wb"))
    plt.figure()
    plt.plot(np.arange(1, MLP.max_num_iter + 1), MLP.train_loss,
             label='train loss')
    plt.plot(np.arange(MLP.valid_epoch, MLP.max_num_iter + 1,
                       step=MLP.valid_epoch, ), MLP.valid_loss,
             label='valid loss')
    plt.xlabel('iteration[number]')
    plt.ylabel('loss[softmax cross entropy]')
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True)
    plt.savefig('.\FigDummy' + print_name + '_Train.pdf', format='pdf')

MLP.test()
print('Accuracy: %{}'.format(100 * MLP.acc[0]))
