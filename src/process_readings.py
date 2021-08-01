import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal
from os import walk


# Write helper functions here
def foo(bar):
    return bar


def save_to_audio_file(filename, data, data_freq):
    import wave
    import struct
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(1)
        f.setframerate(data_freq * 2)
        for sample in data:
            f.writeframes(struct.pack("<h", int(sample)))


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
    body = 'jaw_shake'

    f = []
    for (dirpath, dirnames, filenames) in walk(f'../raw_data/{body}'):
        #filenames.extend(filenames)
        pass


    for filename in filenames:
        df = pd.read_csv(f'../raw_data/{body}/{filename}')
        df = fix_boundary_problem(df)

        # basic processing
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
        fig.suptitle('Accelerometer Data, raw vs. processed')
        axs[0].plot(df.index / data_freq, df.acc_x)
        axs[1].plot(df.index / data_freq, df.acc_y)
        axs[2].plot(df.index / data_freq, df.acc_z)

        # apply filter
        sos = scipy.signal.butter(4, 20, 'high', fs=1000, output='sos')
        if apply_filter:
            df.acc_x = scipy.signal.sosfilt(sos, df.acc_x)
            df.acc_y = scipy.signal.sosfilt(sos, df.acc_y)
            df.acc_z = scipy.signal.sosfilt(sos, df.acc_z)

        axs[0].plot(df.index / data_freq, df.acc_x, label='new')
        axs[1].plot(df.index / data_freq, df.acc_y, label='new')
        axs[2].plot(df.index / data_freq, df.acc_z, label='new')
        df.to_csv(f'../processed_data/imu_data_filtered/{body}/{body + time_display}.csv')
        # fig.show(block=False)
        #plt.savefig(f'../processed_data/{filename[0:8]}.png', dpi=300, bbox_inches='tight')

        fig2, axs2 = plt.subplots(3, sharex=True, sharey=True)
        fig2.suptitle('Accelerometer Data, processed and trimmed')
        axs2[0].plot(df[100:].index / data_freq, df[100:].acc_x)
        axs2[1].plot(df[100:].index / data_freq, df[100:].acc_y)
        axs2[2].plot(df[100:].index / data_freq, df[100:].acc_z)
        # fig2.show(block=False)

        # todo: combine 3 axis data to one, take z axis data for now
        # compute the Fast Fourier Transform (FFT)
        f = np.array(df[100:].acc_z)
        # f_amp = [0 if abs(x) < 20 else x for x in f]

        dt = 1 / data_freq
        n = len(f)
        t = np.arange(0, n / data_freq, dt)

        fhat = np.fft.fft(f, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1/(dt * n)) * np.arange(n)
        L = np.arange(1, np.floor(n/2), dtype='int')

        # fhat_amp = np.fft.fft(f_amp, n)
        # PSD_amp = fhat_amp * np.conj(fhat_amp) / n

        fig3, axs3 = plt.subplots(2,1)

        plt.sca(axs3[0])
        plt.plot(t, f, color='c')
        plt.xlim(t[0], t[-1])

        plt.sca(axs3[1])
        plt.plot(freq[L], PSD[L], color='c')
        plt.xlim(freq[L[0]], freq[L[-1]])

        # Use the PSD to filter out noise
        indices = PSD > 2000
        PSDclean = PSD * indices
        fhat = indices * fhat
        ffilt = np.fft.ifft(fhat)

        fig4, axs4 = plt.subplots(3, 1)

        plt.sca(axs4[0])
        plt.plot(t, f, color='c', linewidth=1.5, label='raw_data')
        # plt.plot(t, f_amp, color='r', linewidth=2, label='amp')
        plt.xlim(t[0], t[-1])
        plt.legend()

        plt.sca(axs4[1])
        plt.plot(t, f, color='k', linewidth=2, label='filtered_data')
        plt.xlim(t[0], t[-1])
        plt.legend()

        plt.sca(axs4[2])
        plt.plot(freq[L], PSD[L], color='c', linewidth=2, label='raw')
        # plt.plot(freq[L], PSD_amp[L], color='r', linewidth=2, label='amp')
        plt.plot(freq[L], PSDclean[L], color='k', linewidth=1.5, label='filtered')
        plt.xlim(freq[L[0]], freq[L[-1]])
        plt.legend()

        plt.savefig(f'../processed_data/{filename[0:10]}.png', dpi=300, bbox_inches='tight')
    plt.show()