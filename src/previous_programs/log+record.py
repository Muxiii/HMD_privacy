import serial
import time
import re
import pandas as pd
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
body = 'jaw'

def parse_readings(text: str):
    """ Parse IMU output string into a list of floats.

    :param text: IMU output string containing Accelerometer, Gyroscope and Magnetometer readings in any order
    :return: A list of float in the same order
    """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    result_str = rx.findall(text)
    return [float(x) for x in result_str]


if __name__ == '__main__':
    """ 
    Run this script to log serial port readings into a csv with desired format. csv files will be saved in 
    'raw data' folder with timestamp as filename. 
    """

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    acc_x = []
    acc_y = []
    acc_z = []
    time_count = []
    realtime = []

    # modify with the correct serial port name and datarate, e.g. 'COM1', 115200
    ser = serial.Serial('COM3', 230400, timeout=2)

    ser.flushInput()
    ser.flushOutput()
    start_time = time.time()
    first_reading = None
    line = str(ser.readline())

    print("* recording")
    frames = []
    t_last = time.time()

    # modify with the desired recording length
    while time.time() - start_time <= RECORD_SECONDS:
    #for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # bytesToRead = ser.inWaiting()
        # ser.read(bytesToRead)
        t = time.time()
        if t -t_last >= CHUNK / RATE:
            data = stream.read(CHUNK)
            frames.append(data)
            t_last = t

        line = str(ser.readline())
        parsed_data = parse_readings(line)

        acc_x.append(parsed_data[0])
        acc_y.append(parsed_data[1])
        acc_z.append(parsed_data[2])
        time_count.append(parsed_data[3])
        realtime.append(time.time())

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    time_display = str(time.localtime().tm_hour)+';'+str(time.localtime().tm_min)+';'+str(time.localtime().tm_sec)
    WAVE_OUTPUT_FILENAME = body + time_display + '.wav'
    print(time_display)

    wf = wave.open(f'../raw_data/voice_record/{body}/{WAVE_OUTPUT_FILENAME}', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # convert into pandas Dataframe object and save as csv
    df = pd.DataFrame({
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "time_count" : time_count,
        "realtime" : realtime
    })

    df.to_csv(f'../raw_data/imu_data/{body}/{body + time_display}.csv')