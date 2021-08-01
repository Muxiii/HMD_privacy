import serial
import time
import re
import pandas as pd


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
    acc_x = []
    acc_y = []
    acc_z = []

    # modify with the correct serial port name and datarate, e.g. 'COM1', 115200
    ser = serial.Serial('COM3', 230400, timeout=2)

    ser.flushInput()
    ser.flushOutput()
    start_time = time.time()
    first_reading = None
    line = str(ser.readline())

    # modify with the desired recording length
    while time.time() - start_time <=5:
        # bytesToRead = ser.inWaiting()
        # ser.read(bytesToRead)
        line = str(ser.readline())

        parsed_data = parse_readings(line)
        acc_x.append(parsed_data[0])
        acc_y.append(parsed_data[1])
        acc_z.append(parsed_data[2])

    # convert into pandas Dataframe object and save as csv
    df = pd.DataFrame({
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
    })

    df.to_csv(f'../raw_data/jaw_shake/{time.time()}.csv')