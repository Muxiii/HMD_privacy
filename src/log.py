from threading import Thread, Barrier
import time
import serial
import re
import pandas as pd
import pyaudio
import wave
import scipy.signal
import os
from os import walk
from scipy.io import wavfile
import conf

""" 
    PARAMETERS to set:
    RECORD_SECONDS : final duration of the audio
    adjust_factor : the 2 logging threadings (audio & imu) are set to be synchronized.
                    this factor is to compensate for the unknown ti  me lag of imu-logging
    extend_factor : the audio record should be a little longer than desinated, and redundant part would be abandoned
    body : the location of the IMU sensor
    
"""
RECORD_SECONDS = conf.SECONDS
adjust_factor = 0.18
extend_factor = 0.002
RECORD_SECONDS_audio = RECORD_SECONDS + adjust_factor + extend_factor
body = 'jaw_f_p'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

TIME_DISPLAY = str(time.localtime().tm_mon) + str(time.localtime().tm_mday) + ' ; ' +str(time.localtime().tm_hour) \
               + '_' + str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec)
WAVE_OUTPUT_FILENAME = body +' ; ' + TIME_DISPLAY + '.wav'
CSV_OUTPUT_FILENAME = body +' ; ' + TIME_DISPLAY + '.csv'


def parse_readings(text: str):
    """ Parse IMU output string into a list of floats.

    :param text: IMU output string containing Accelerometer, Gyroscope and Magnetometer readings in any order
    :return: A list of float in the same order
    """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    result_str = rx.findall(text)
    return [float(x) for x in result_str]


def threading_1_imu():
    """ log IMU data """
    acc_x = []
    acc_y = []
    acc_z = []
    time_count = []
    realtime = []

    """modify with the correct serial port name and datarate, e.g. 'COM1', 115200"""
    ser = serial.Serial('COM3', 230400, timeout=2)
    time.sleep(0.1)
    ser.flushInput()
    ser.flushOutput()

    all_process_ready.wait()    # do not start until all processes are ready
    ser.flushInput()
    ser.flushOutput()
    print('* recording imu')
    _ = ser.readline()

    start_time = time.time()
    # modify with the desired recording length
    while time.time() - start_time <= RECORD_SECONDS + 0.1:
        # bytesToRead = ser.inWaiting()
        # ser.read(bytesToRead)

        line = str(ser.readline())
        parsed_data = parse_readings(line)

        acc_x.append(parsed_data[0])
        acc_y.append(parsed_data[1])
        acc_z.append(parsed_data[2])
        time_count.append(parsed_data[3])
        realtime.append(time.time())

    print("* done recording imu")

    # convert into pandas Dataframe object and save as csv
    df = pd.DataFrame({
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "time_count" : time_count,
        "realtime" : realtime
    })

    df.to_csv(f'../dataset/{body}/raw_data/imu_data/{CSV_OUTPUT_FILENAME}')
    process(df)

def threading_2_audio():
    """ record audio"""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    frames = []
    _ = stream.read(CHUNK)

    all_process_ready.wait()    # do not start until all processes are ready

    print("* recording audio : ", time.time())

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS_audio)+1):
       data = stream.read(CHUNK)
       frames.append(data)

    print("* done recording audio", time.time())

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(f'../dataset/{body}/raw_data/audio_record/{WAVE_OUTPUT_FILENAME}', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    like = wavfile.read(f'../dataset/{body}/raw_data/audio_record/{WAVE_OUTPUT_FILENAME}')
    wavfile.write(f'../dataset/{body}/raw_data/audio_record/{WAVE_OUTPUT_FILENAME}', RATE,
                  like[1][int(adjust_factor * RATE) : int(RECORD_SECONDS_audio * RATE)])

    #   delete *.pkf file
    wavf = []
    for (dirpath, dirnames, filenames) in walk(f'../dataset/{body}/processed_data/speech_reconstruction/'):
        wavf.extend(filenames)
        break
    for filename in filenames:
        if filename.endswith('.pkf'):
            os.remove(f'../dataset/{body}/processed_data/speech_reconstruction/{filename}')

def process(df):

    """ POST-PROCESSING:
        fix boundary problems
        linear interlolate
        high-pass filtering (20Hz)"""

    # fix issues when raw readings exceeds [0, 4096] they overflow
    df.acc_x = df.acc_x - df.acc_x[0]
    df.acc_y = df.acc_y - df.acc_y[0]
    df.acc_z = df.acc_z - df.acc_z[0]
    #df.time_count = df.time_count - df.time_count[0]
    df.acc_x = df.acc_x.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)
    df.acc_y = df.acc_y.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)
    df.acc_z = df.acc_z.apply(lambda x: x - 4096 if x > 3000 else x + 4096 if x < -3000 else x)

    #linear interpolate
    df.time_count = pd.to_timedelta(df.time_count, unit='us')
    d = df.set_index('time_count')
    t = d.index
    freq_str = str(round(1000 / data_freq, 2))+'ms'
    r = pd.timedelta_range(t.min(), t.max(), freq=freq_str)
    d = d.reindex(t.union(r)).interpolate('index').loc[r]

    # apply filter
    sos = scipy.signal.butter(4, high_pass_threshold, 'high', fs=data_freq, output='sos')
    d.acc_x = scipy.signal.sosfilt(sos, d.acc_x)
    d.acc_y = scipy.signal.sosfilt(sos, d.acc_y)
    d.acc_z = scipy.signal.sosfilt(sos, d.acc_z)

    d.to_csv(f'../dataset/{body}/processed_data/imu_data_processed/p_{CSV_OUTPUT_FILENAME}')
    return df

if __name__ == '__main__':

    data_freq = conf.DATA_FREQ
    high_pass_threshold = conf.HIGH_PASS
    all_process_ready = Barrier(2)
    thread1 = Thread(target=threading_1_imu)
    thread1.start()
    thread2 = Thread(target=threading_2_audio)
    thread2.start()
