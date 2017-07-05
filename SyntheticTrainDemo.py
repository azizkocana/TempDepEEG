from pylab import *
import scipy.signal as ssig
from spectrum import aryule
from TrainingSet import TrainRNN, TrainMLP, TrainOneStepRNN
import numpy \
    as np
import matplotlib.pyplot as plt
import scipy.io as sio


def generate_dummy_sequence(num_samples=100, num_AR_coe=10, num_trials=10,
                            var_gam=0.4 * 128, var_gau=0.1, var_AR=0.3):
    sample_trial = sio.loadmat(
        '.\DaqMat\sample_13a2a_S2.mat')['trialData']
    sample_target = sio.loadmat('.\DaqMat\sample_13a2a_S2.mat')[
        'trialTargetness']

    len_trial = sample_trial.shape[0]
    len_overlap = np.floor(len_trial / 3)
    len_sig = (num_trials - 1) * len_overlap + len_overlap * 5

    target = np.mean(sample_trial[:, 5:9, np.where(sample_target == 1)[1]], 2)
    non_target = sample_trial[:, 5:9, np.where(sample_target == 0)[1]]

    a = np.zeros(num_AR_coe)

    for ar_idx in range(non_target.shape[2]):
        a += aryule(np.transpose(non_target[:, :, ar_idx])[0], 10)[0]
    a = a / ar_idx

    dummy_samples, dummy_labels, dummy_targets = [], [], []
    for smp_idx in range(num_samples):
        tar = np.random.randint(num_trials - 1)
        t = tar * len_overlap
        s = np.sqrt(var_AR) * np.random.random([int(non_target.shape[1]),
                                                int(len_sig)])

        s = ssig.upfirdn(a, s)
        shift_sig = t + np.floor(np.random.gamma(1, sqrt(var_gam)))
        s[:, int(shift_sig): int(shift_sig + len_trial)] += \
            (1 + var_gau * np.random.random(1)) * np.transpose(target)

        l = np.zeros([1, s.shape[1]])
        l[0][int(t):int(t + len_trial)] = np.ones([1, int(len_trial)])
        dummy_samples.append(s.astype(np.float32))
        dummy_labels.append(l.astype(np.int32))

        tmp = np.zeros([1, num_trials])
        tmp[0][tar] = 1
        dummy_targets.append(tmp.astype(np.int32))

    dummy_samples = np.transpose(np.asarray(dummy_samples), (0, 2, 1))
    dummy_labels = np.asarray(dummy_labels)
    dummy_targets = np.asarray(dummy_targets)

    return dummy_samples, dummy_labels, dummy_targets


""" Data should be three dimensional array
    data: [num_samples x length of time signal x number of channels]
    label: [num_samples x 1 x length of time signal]"""
# samples_o = sio.loadmat('data.mat')
# labels_o = sio.loadmat('label.mat')
# raw_samples = samples_o['data_m']
# raw_labels = np.concatenate(labels_o['label_m']).astype(np.int32)

folder_name = '\inst3'
print_name = 'sample_13a2a_S2'
mode = 'RNN'

num_samples = 300
num_AR_coe = 10
num_trials = 10
mag_shift_target = 0.4

max_num_iter = 100
pow_non_target = [20]
shift_offset_target = list(np.asarray([0.001]) * 128 * 0.1)

RNN, MLP, OneStepRNN = [], [], []
error = np.ones([int(len(pow_non_target)), int(len(shift_offset_target))])
idx = 0
for idx_x in range(len(pow_non_target)):
    for idx_y in range(len(shift_offset_target)):
        raw_samples, raw_labels, raw_targets = generate_dummy_sequence(
            num_samples=num_samples, num_AR_coe=num_AR_coe,
            num_trials=num_trials, var_gam=shift_offset_target[idx_y],
            var_gau=mag_shift_target,
            var_AR=pow_non_target[idx_x])

        if mode == 'RNN':
            raw_labels = np.concatenate(raw_labels)
            RNN.append(TrainRNN(raw_samples, raw_labels,
                                max_num_iter=max_num_iter, num_layer=5))
            RNN[idx].train()
            RNN[idx].test()

            plt.figure()
            plt.plot(np.arange(1, RNN[idx].max_num_iter + 1),
                     RNN[idx].train_loss,
                     label='train loss')
            plt.plot(
                np.arange(RNN[idx].valid_epoch, RNN[idx].max_num_iter + 1,
                          step=RNN[0].valid_epoch,print_name=print_name),
                RNN[idx].valid_loss, label='valid loss')
            plt.xlabel('iteration[number]')
            plt.ylabel('loss[softmax cross entropy]')
            plt.legend(loc='upper right', shadow=True)
            plt.grid(True)
            plt.savefig('.\FigDummy\inst3\idx_' + str(idx) + '_Train.pdf',
                        format='pdf', )

            error[idx_x, idx_y] = RNN[idx].test_loss[0]

            idx += 1

        if mode == 'MLP':
            raw_labels = np.concatenate(raw_targets)

            MLP.append(TrainMLP(raw_samples, raw_labels,
                                max_num_iter=max_num_iter, num_layer=5))
            MLP[idx].train()
            MLP[idx].test()

            plt.figure()
            plt.plot(np.arange(1, MLP[idx].max_num_iter + 1),
                     MLP[idx].train_loss,
                     label='train loss')
            plt.plot(
                np.arange(MLP[idx].valid_epoch, MLP[idx].max_num_iter + 1,
                          step=MLP[0].valid_epoch),
                MLP[idx].valid_loss, label='valid loss')
            plt.title('Accuracy: %{}'.format(100 * MLP[idx].acc[0]))
            plt.xlabel('iteration[number]')
            plt.ylabel('loss[softmax cross entropy]')
            plt.legend(loc='upper right', shadow=True)
            plt.grid(True)
            plt.savefig('.\FigDummy' + folder_name + '\idx_' + str(idx) +
                        '_Train.pdf',
                        format='pdf')

            error[idx_x, idx_y] = MLP[idx].test_loss[0]

            idx += 1

        if mode == 'OneStepRNN':
            raw_labels = np.concatenate(raw_targets)

            OneStepRNN.append(TrainOneStepRNN(raw_samples, raw_labels,
                                              max_num_iter=max_num_iter))
            OneStepRNN[idx].train()
            OneStepRNN[idx].test()

            plt.figure()
            plt.plot(np.arange(1, OneStepRNN[idx].max_num_iter + 1),
                     OneStepRNN[idx].train_loss,
                     label='train loss')
            plt.plot(
                np.arange(OneStepRNN[idx].valid_epoch,
                          OneStepRNN[idx].max_num_iter + 1,
                          step=OneStepRNN[0].valid_epoch),
                OneStepRNN[idx].valid_loss, label='valid loss')
            plt.title('Accuracy: %{}'.format(100 * OneStepRNN[idx].acc[0]))
            plt.xlabel('iteration[number]')
            plt.ylabel('loss[softmax cross entropy]')
            plt.legend(loc='upper right', shadow=True)
            plt.grid(True)
            plt.savefig('.\FigDummy' + folder_name + '\idx_' + str(idx) +
                        '_Train.pdf',
                        format='pdf')

            error[idx_x, idx_y] = OneStepRNN[idx].test_loss[0]

            idx += 1

if mode == 'RNN':
    plt.figure()
    fig = plt.imshow(error, cmap='gray', interpolation='None')
    cbr = plt.colorbar(fig)
    plt.title('softmax cross entropy')
    plt.ylabel('pow_non_target')
    plt.xlabel('shift_offset')
    plt.savefig('.\FigDummy\inst3\Error_Test.pdf', format='pdf')

    text_file = open(".\FigDummy" + folder_name + "\params.txt", "w")
    text_file.write("non target variance: {} - shift onset: {}".format(
        np.asarray(pow_non_target),
        np.asarray(shift_offset_target).astype(np.int32)))
    text_file.close()

if mode == 'MLP':
    plt.figure()
    fig = plt.imshow(error, cmap='gray', interpolation='None')
    cbr = plt.colorbar(fig)
    plt.title('softmax cross entropy')
    plt.ylabel('pow_non_target')
    plt.xlabel('shift_offset')
    plt.savefig('.\FigDummy' + folder_name + '\Error_Test.pdf', format='pdf')

    text_file = open('.\FigDummy' + folder_name + '\params.txt', "w")
    text_file.write("non target variance: {} - shift onset: {}".format(
        np.asarray(pow_non_target),
        np.asarray(shift_offset_target).astype(np.int32)))
    text_file.close()

if mode == 'OneStepRNN':
    plt.figure()
    fig = plt.imshow(error, cmap='gray', interpolation='None')
    cbr = plt.colorbar(fig)
    plt.title('softmax cross entropy')
    plt.ylabel('pow_non_target')
    plt.xlabel('shift_offset')
    plt.savefig('.\FigDummy' + folder_name + '\Error_Test.pdf', format='pdf')

    text_file = open('.\FigDummy' + folder_name + '\params.txt', "w")
    text_file.write("non target variance: {} - shift onset: {}".format(
        np.asarray(pow_non_target),
        np.asarray(shift_offset_target).astype(np.int32)))
    text_file.close()
