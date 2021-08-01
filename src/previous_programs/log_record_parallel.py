from multiprocessing import Process
import time
import serial
import re
import pandas as pd
import pyaudio
import wave

RECORD_SECONDS = 15
body = 'jaw'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

TIME_DISPLAY = str(time.localtime().tm_hour) + ';' + str(time.localtime().tm_min) + ';' + str(time.localtime().tm_sec)
WAVE_OUTPUT_FILENAME = body + TIME_DISPLAY + '.wav'
#print(TIME_DISPLAY)


def parse_readings(text: str):
    """ Parse IMU output string into a list of floats.

    :param text: IMU output string containing Accelerometer, Gyroscope and Magnetometer readings in any order
    :return: A list of float in the same order
    """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    result_str = rx.findall(text)
    return [float(x) for x in result_str]


def process1_imu():
    """subprocess1 : log IMU data for n seconds"""
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

    line = str(ser.readline())

    p2 = Process(target=peocess2_audio)
    p2.start()
    print('* recording imu')
    start_time = time.time()
    # modify with the desired recording length
    while time.time() - start_time <= RECORD_SECONDS:
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

    df.to_csv(f'../raw_data/imu_data/{body}/{body + TIME_DISPLAY}.csv')


def peocess2_audio():
    """subprocess2 : record AUDIO for n seconds"""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording audio")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
       data = stream.read(CHUNK)
       frames.append(data)

    print("* done recording audio")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(f'../raw_data/voice_record/{body}/{WAVE_OUTPUT_FILENAME}', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':
    p1 = Process(target=process1_imu)
    p1.start()
