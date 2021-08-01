import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal
from os import walk
import wave
import struct
import noisereduce as nr
import soundfile as sf

# Write helper functions here
def foo(bar):
    return bar


def save_to_audio_file(filename, data, data_freq):
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
    body = 'jaw'

    f = []
    for (dirpath, dirnames, filenames) in walk(f'../raw_data/imu_data/{body}/'):
        f.extend(filenames)
        break

    for filename in f:
        df = pd.read_csv(f'../raw_data/imu_data/{body}/{filename}')
        label = filename.split('_')[0]

        """Adjust data rate"""
        f_index = np.array(df.index)
        f_time = np.array(df.time_count)
        data_freq = int(f_index[-1] * 1000000 / f_time[-1])+1

        # basic processing
        df = fix_boundary_problem(df)

        # apply filter
        sos = scipy.signal.butter(4, 20, 'high', fs=1000, output='sos')
        if apply_filter:
            df.acc_x = scipy.signal.sosfilt(sos, df.acc_x)
            df.acc_y = scipy.signal.sosfilt(sos, df.acc_y)
            df.acc_z = scipy.signal.sosfilt(sos, df.acc_z)

        # compute the Fast Fourier Transform (FFT)
        f_x = np.array(df[100:].acc_x)
        f_y = np.array(df[100:].acc_y)
        f_z = np.array(df[100:].acc_z)

        # normalize
        f_x = f_x / max(f_x)
        f_y = f_y / max(f_y)
        f_z = f_z / max(f_z)

        sf.write(f'../processed_data/speech_reconstruction/{body}/{filename[:-4]}_x.wav', f_x, data_freq)
        sf.write(f'../processed_data/speech_reconstruction/{body}/{filename[:-4]}_y.wav', f_y, data_freq)
        sf.write(f'../processed_data/speech_reconstruction/{body}/{filename[:-4]}_z.wav', f_z, data_freq)


    # use noisereduce package
    wavf = []
    for (dirpath, dirnames, filenames) in walk(f'../processed_data/speech_reconstruction/{body}/'):
        wavf.extend(filenames)
        break

    for filename in wavf:
        # load data
        data, rate = sf.read(f'../processed_data/speech_reconstruction/{body}/{filename}')
        # choose noise_clip

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate, freq_mask_smooth_hz=160,
                                        time_mask_smooth_ms=int(256 * 1000 / data_freq) +15 , stationary=True)

        fig, axs = plt.subplots(1, 1)
        plt.sca(axs)
        plt.title(filename)
        plt.plot(data, label='raw')
        plt.plot(reduced_noise, label='reduced')
        plt.legend()

        sf.write(f'../processed_data/speech_reconstruction/{body}/{filename[:-4]}_reduced.wav', reduced_noise/max(reduced_noise), rate)

    plt.show()
