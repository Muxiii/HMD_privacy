import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal
from os import walk


# Write helper functions here
def foo(bar):
    return bar


def fix_boundary_problem(df):
    # fix issues when raw readings exceeds [0, 4096] they overflow
    df.acc_x = df.acc_x - df.acc_x[0]
    df.acc_y = df.acc_y - df.acc_y[0]
    df.acc_z = df.acc_z - df.acc_z[0]
    df.acc_x = df.acc_x.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)
    df.acc_y = df.acc_y.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)
    df.acc_z = df.acc_z.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)

    return df


if __name__ == '__main__':
    # Parameters
    apply_filter = True
    data_freq = 1000
    body = 'ear'

    f = []
    for (dirpath, dirnames, filenames) in walk(f'../raw_data/{body}/'):
        f.extend(filenames)
        break

    for filename in f:
        df = pd.read_csv(f'../raw_data/{body}/{filename}')
        label = filename.split('_')[0]

        # basic processing
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
        fig.suptitle(f'Label {label} Accelerometer Data, raw vs. processed')
        axs[0].plot(df.index / data_freq, df.acc_x)
        axs[1].plot(df.index / data_freq, df.acc_y)
        axs[2].plot(df.index / data_freq, df.acc_z)

        df = fix_boundary_problem(df)

        # apply filter
        sos = scipy.signal.butter(4, 20, 'high', fs=1000, output='sos')
        if apply_filter:
            df.acc_x = scipy.signal.sosfilt(sos, df.acc_x)
            df.acc_y = scipy.signal.sosfilt(sos, df.acc_y)
            df.acc_z = scipy.signal.sosfilt(sos, df.acc_z)

        axs[0].plot(df.index / data_freq, df.acc_x, label='new')
        axs[1].plot(df.index / data_freq, df.acc_y, label='new')
        axs[2].plot(df.index / data_freq, df.acc_z, label='new')
        plt.savefig(f'../processed_data/{body}/{filename}_raw_vs_high_pass.png', dpi=300, bbox_inches='tight')
        # fig.show(block=False)

        fig2, axs2 = plt.subplots(3, sharex=True, sharey=True)
        fig2.suptitle(f'Label {label} Accelerometer Data, high pass')
        axs2[0].plot(df[100:].index / data_freq, df[100:].acc_x)
        axs2[1].plot(df[100:].index / data_freq, df[100:].acc_y)
        axs2[2].plot(df[100:].index / data_freq, df[100:].acc_z)
        plt.savefig(f'../processed_data/{body}/{filename}_high_pass_only.png', dpi=300, bbox_inches='tight')
        # fig2.show(block=False)

        # todo: combine 3 axis data to one, take z axis data for now
        # compute the Fast Fourier Transform (FFT)
        f_x = np.array(df[100:].acc_x)
        f_y = np.array(df[100:].acc_y)
        f_z = np.array(df[100:].acc_z)
        # f_amp = [0 if abs(x) < 20 else x for x in f]

        dt = 1 / data_freq
        n = len(f_x)
        t = np.arange(0, n / data_freq, dt)

        fhat_x = np.fft.fft(f_x, n)
        PSD_x = fhat_x * np.conj(fhat_x) / n
        fhat_y = np.fft.fft(f_y, n)
        PSD_y = fhat_y * np.conj(fhat_y) / n
        fhat_z = np.fft.fft(f_z, n)
        PSD_z = fhat_z * np.conj(fhat_z) / n
        freq = (1/(dt * n)) * np.arange(n)
        L = np.arange(1, np.floor(n/2), dtype='int')

        PSD_max = np.max(np.stack((PSD_x, PSD_y, PSD_z)), axis=0)
        peaks = scipy.signal.find_peaks(PSD_max[L], height=np.max(PSD_max)/3, distance=300)

        fig3, axs3 = plt.subplots(1, 1)

        plt.sca(axs3)
        plt.title(f'Label {label} FFT w/ peaks {freq[peaks[0][0:3]]}')
        plt.plot(freq[L], PSD_max[L], color='c', linewidth=2)
        plt.xlim(freq[L[0]], freq[L[-1]])

        plt.savefig(f'../processed_data/{body}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()