import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def trigger_decoder(x, trigger_partitioner):
    """ Decodes RSVPKeyboard stored data using the trigger partitioner
        Args:
            x(matlab_array): Nx1 array denoting triggers
            trigger_partitioner(dictionary): Stores trigger information.
            struct data type from MATLAB is converted into dictionary in py.

    """

    # x = sio.loadmat('.\DaqMat\dummytrig.mat')
    # x = x['afterFrontendFilterTrigger']
    # x = x.reshape(1, len(x))[0]
    # x = x.astype(np.int32)

    # TODO: Implement trigger decoder afterwards. Assume it is known now.
    len_win = trigger_partitioner['windowLengthinSamples']
    id_sequence_end = trigger_partitioner['sequenceEndID']
    id_pause = trigger_partitioner['pauseID']
    id_fixation = trigger_partitioner['fixationID']
    offset_trig = trigger_partitioner['TARGET_TRIGGER_OFFSET']

    # Find the falling edges and labels of the trials
    x = np.asarray(x)
    x = x.reshape(1, len(x))[0]
    x = x.astype(np.int32)
    time_last = len(x) - len_win
    x = x[0:time_last]
    time_fall_edge = np.where(np.diff(x) < 0)[0] - 1
    labels_x = x[time_fall_edge]

    # TODO: check for paused sequences and discard them

    # Calculate number of sequences
    num_finished_seq = np.sum(labels_x == id_sequence_end)
    idx_fix = np.array(np.where(labels_x == id_fixation))[0]
    idx_end_seq = np.array(np.where(labels_x == id_sequence_end))[0]
    # TODO: Shapes of both idx should be same

    # Decompose labels into sequences
    labels_seq, true_seq, target_seq, timing_seq = [], [], [], []
    for i in range(len(idx_end_seq)):
        labels_seq.append(labels_x[idx_fix[i] + 1:idx_end_seq[i]])
        timing_seq.append(time_fall_edge[idx_fix[i] + 1:idx_end_seq[i]])

    # Number different triggers in a sequence
    len_sequence = len(labels_x) / num_finished_seq

    # Check if the mode is calibration
    if idx_fix[0] != 0:
        for i in range(len(idx_end_seq)):
            true_seq.append(labels_x[idx_fix[i] - 1] - offset_trig)
            target_seq.append((labels_seq[i] == true_seq[i]).astype(np.int32))
    else:
        true_seq = [np.zeros(
            int(idx_end_seq[0] - (idx_fix[0] + 1)))] * num_finished_seq
        target_seq = [np.zeros(
            int(idx_end_seq[0] - (idx_fix[0] + 1)))] * num_finished_seq

    labels_seq = np.array(labels_seq)
    timing_seq = np.array(timing_seq)
    true_seq = np.array(true_seq)
    target_seq = np.array(target_seq)

    return [labels_seq, timing_seq, true_seq, target_seq]
