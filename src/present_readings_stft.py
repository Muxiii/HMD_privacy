import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal
from os import walk


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
    body = 'above_nose'

    f = []
    for (dirpath, dirnames, filenames) in walk(f'../raw_data/{body}/'):
        f.extend(filenames)
        break

    for filename in f:
        df = pd.read_csv(f'../raw_data/{body}/{filename}')
        label = filename.split('_')[0]

        df = fix_boundary_problem(df)

        # apply filter
        sos = scipy.signal.butter(4, 20, 'high', fs=1000, output='sos')
        if apply_filter:
            df.acc_x = scipy.signal.sosfilt(sos, df.acc_x)
            df.acc_y = scipy.signal.sosfilt(sos, df.acc_y)
            df.acc_z = scipy.signal.sosfilt(sos, df.acc_z)


        # todo: combine 3 axis data to one, take z axis data for now
        # compute the Fast Fourier Transform (FFT)
        f_x = np.array(df[100:].acc_x)
        f_y = np.array(df[100:].acc_y)
        f_z = np.array(df[100:].acc_z)
        # f_amp = [0 if abs(x) < 20 else x for x in f]

        dt = 1 / data_freq
        n = len(f_x)

        xf, xt, xZxx = scipy.signal.stft(f_x, data_freq, nperseg=256, noverlap=240)
        yf, yt, yZxx = scipy.signal.stft(f_y, data_freq, nperseg=256, noverlap=240)
        zf, zt, zZxx = scipy.signal.stft(f_y, data_freq, nperseg=256, noverlap=240)

        plt.pcolormesh(xt, xf, np.abs(xZxx), shading='gouraud')
        plt.title(f'Label {label} STFT Magnitude, x axis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(f'../processed_data/stft/{body}/{filename[0:10]}_x.png', dpi=300, bbox_inches='tight')

        plt.pcolormesh(yt, yf, np.abs(yZxx), shading='gouraud')
        plt.title(f'Label {label} STFT Magnitude, y axis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(f'../processed_data/stft/{body}/{filename[0:10]}_y.png', dpi=300, bbox_inches='tight')

        plt.pcolormesh(zt, zf, np.abs(zZxx), shading='gouraud')
        plt.title(f'Label {label} STFT Magnitude, z axis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(f'../processed_data/stft/{body}/{filename[0:10]}_z.png', dpi=300, bbox_inches='tight')
    plt.show()