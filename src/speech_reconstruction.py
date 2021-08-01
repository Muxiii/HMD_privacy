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

if __name__ == '__main__':
    # Parameters
    apply_filter = True
    body = 'jaw_f_p'
    DATA_FREQ = 1000

    f = []
    for (dirpath, dirnames, filenames) in walk(f'../dataset/{body}/processed_data/imu_data_processed'):
        f.extend(filenames)
        break

    for filename in f:
        d = pd.read_csv(f'../dataset/{body}/processed_data/imu_data_processed/{filename}')
        label = filename.split('_')[0]

        # compute the Fast Fourier Transform (FFT)
        f_x = np.array(d[100:].acc_x)
        f_y = np.array(d[100:].acc_y)
        f_z = np.array(d[100:].acc_z)

        # normalize
        f_x = f_x / max(f_x)
        f_y = f_y / max(f_y)
        f_z = f_z / max(f_z)

        sf.write(f'../dataset/{body}/processed_data/speech_reconstruction/{filename[:-4]}_x.wav', f_x, DATA_FREQ)
        sf.write(f'../dataset/{body}/processed_data/speech_reconstruction/{filename[:-4]}_y.wav', f_y, DATA_FREQ)
        sf.write(f'../dataset/{body}/processed_data/speech_reconstruction/{filename[:-4]}_z.wav', f_z, DATA_FREQ)


    # use noisereduce package
    wavf = []
    for (dirpath, dirnames, filenames) in walk(f'../dataset/{body}/processed_data/speech_reconstruction/'):
        wavf.extend(filenames)
        break

    for filename in wavf:

        # avoid dealing with processed audio
        if filename.endswith('reduced.wav') :
            continue

        # load data
        data, rate = sf.read(f'../dataset/{body}/processed_data/speech_reconstruction/{filename}')

        # choose noise_clip

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate, freq_mask_smooth_hz=160,
                                        time_mask_smooth_ms=300, stationary=True)

        fig, axs = plt.subplots(1, 1)
        plt.sca(axs)
        plt.title(filename)
        plt.plot(data, label='raw')
        plt.plot(reduced_noise, label='reduced')
        plt.legend()

        sf.write(f'../dataset/{body}/processed_data/speech_reconstruction/{filename[:-4]}_reduced.wav',
                 reduced_noise/max(reduced_noise), rate)

    plt.show()
