from pydub import AudioSegment
from os import walk
import conf
import pandas as pd
import numpy as np

def get_len(wav_path):
    """
    return the duration of deignated file (in sec)
    """

    sound = AudioSegment.from_mp3(wav_path)
    return( len(sound) / 1000.0)

def get_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    """
    cut deignated audio file into 1 fragment
    :param main_wav_path: 原音频文件路径
    :param start_time: 截取的开始时间
    :param end_time: 截取的结束时间
    :param part_wav_path: 截取后的音频路径
    :return:
    """
    start_time = start_time * data_freq
    end_time = end_time * data_freq

    sound = AudioSegment.from_mp3(main_wav_path)
    segment = sound[start_time:end_time]

    segment.export(part_wav_path, format="wav")


def cut_up_audio():

    f1 = []
    f2 = []

    for (dirpath, dirnames, filenames) in walk(wav_path1):
        f1.extend(filenames)
        break

    for filename in f1:
        audio_length = get_len(wav_path1 + '/' + filename)
        s = 0
        e = step_length
        i = 1
        while e < audio_length:
            fragment_path = cut_wav_path1+'/'+filename[:-4]+'-'+str(i)+'.wav'
            get_part_wav(wav_path1+'/'+filename, s, e, fragment_path)
            s = s + step_length - overlap
            e = e + step_length - overlap
            i = i + 1
        print('Audio fragments:', i - 1)

    for (dirpath, dirnames, filenames) in walk(wav_path2):
        f2.extend(filenames)
        break

    for filename in f2:
        audio_length = get_len(wav_path2 + '/' + filename)
        s = 0
        e = step_length
        i = 1
        while e < audio_length:
            fragment_path = cut_wav_path2 + '/' + filename[:-4] + '-' + str(i) + '.wav'
            get_part_wav(wav_path2 + '/' + filename, s, e, fragment_path)
            s = s + step_length - overlap
            e = e + step_length - overlap
            i = i + 1
        print('R.Audio fragments:', i - 1)


def cut_up_imu():

    f = []

    for (dirpath, dirnames, filenames) in walk(imu_path):
        f.extend(filenames)
        break

    for filename in f:
        df = pd.read_csv(f'{imu_path}/{filename}')
        s = start_time
        e = start_time + step_length
        i = 1
        print('IMU frames: ',len(df))
        while e*data_freq <= len(df):

            _ = df[int(s*data_freq):int(e*data_freq)]
            acc_x = np.array(_.acc_x)
            acc_y = np.array(_.acc_y)
            acc_z = np.array(_.acc_z)

            df_fragment = pd.DataFrame({
                "acc_x": acc_x,
                "acc_y": acc_y,
                "acc_z": acc_z,
            })

            csv_output_filename = filename + '_'+str(i)+'.csv'
            df_fragment.to_csv(f'{cut_imu_path}/{csv_output_filename}')
            s = s + step_length - overlap
            e = e + step_length - overlap
            i = i + 1

        print('IMU fragments:', i-1)


if __name__ == '__main__':

    # import parameters from conf.py
    data_freq = conf.DATA_FREQ
    step_length = conf.FRAGMENT_LENGTH
    overlap = conf.OVERLAP
    body = conf.BODY
    start_time = conf.START_TIME

    # define input & output paths
    wav_path1 = f'../dataset/{body}/raw_data/audio_record'
    cut_wav_path1 = f'../dataset/{body}/processed_data/audio_cuts'
    wav_path2 = f'../dataset/{body}/processed_data/speech_reconstruction'
    cut_wav_path2 = f'../dataset/{body}/processed_data/reconstructed_audio_cuts'
    imu_path = f'../dataset/{body}/processed_data/imu_data_processed'
    cut_imu_path = f'../dataset/{body}/processed_data/imu_cuts'

    cut_up_audio()
    cut_up_imu()