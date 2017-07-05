""" DEMO for print """
import matplotlib.pyplot as plt
import numpy as np
import pickle

FS = 18
FS1 = 15

# train_dat = [valid_epoch, max_num_iter, train_loss, valid_loss[]]
train_dat = pickle.load(open(".\Tmp\TrainPlot.p", "rb"))

# plt.style.use('grayscale')
fig = plt.figure()
sub1 = fig.add_subplot(211)
sub1.plot(np.arange(1, train_dat[1] + 1), train_dat[2], 'o-',
          label='train',
          linewidth=2)
for idx_val_plot in range(5):
    sub1.plot(
        np.arange(train_dat[0], train_dat[1] + 1,
                  step=train_dat[0], ), train_dat[3][idx_val_plot],
        label='valid_' + str(idx_val_plot + 1), linewidth=2)

sub1.set_xlabel('iteration[number]', fontsize=FS)
sub1.set_ylabel('loss[cross entropy]', fontsize=FS)
sub1.legend(loc='upper right', shadow=True, fontsize=FS1)

sub1.spines['top'].set_visible(False)
sub1.spines['right'].set_visible(False)
sub1.get_xaxis().tick_bottom()
sub1.get_yaxis().tick_left()
fig.savefig('.\latex_fig\Train.jpg', format='jpg', bbox_inches='tight',
            pad_inches=0.0, dpi=1200)
plt.show()

# temporal_dat = [x[],y[],p[]]
temporal_dat = pickle.load(open(".\Tmp\TempPlot.p", "rb"))
for j in range(len(temporal_dat[1])):
    x = temporal_dat[0][j]
    y = temporal_dat[1][j]
    p = temporal_dat[2][j]

    tmp = np.array([i for i in range(temporal_dat[0][j].shape[0])]) / 256

    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    ax2 = ax1.twinx()
    ax1.plot(tmp, x, linewidth=2, color='black')
    ax2.plot(tmp, p[:, -1], '.-', label='estimate', linewidth=2)
    ax2.plot(tmp, y, '--', label='true', linewidth=2)

    ax1.set_xlim(0, temporal_dat[0][j].shape[0] / 256)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel('time [s]', fontsize=FS)
    ax1.set_ylabel('data [uV]', fontsize=FS)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('probability', fontsize=FS)
    ax2.legend(loc='upper right', shadow=True, fontsize=FS1)

    ax1.grid(True)

    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_right()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    plt.xticks()
    plt.savefig('.\latex_fig\Temp-S' + str(j) + 'DSFig.jpg',
                format='jpg', bbox_inches='tight', pad_inches=0.0, dpi=1200)
    plt.show()
