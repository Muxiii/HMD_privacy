from pydub import AudioSegment
from os import walk
import pandas as pd

if __name__ == '__main__':

    step_length = 0.5
    overlap = 0
    start_time = 0.1
    data_freq = 1000
    body = 'jaw_f_p'
    imu_path = f'../dataset/{body}/processed_data/imu_data_processed'
    cut_imu_path = f'../dataset/{body}/processed_data/imu_cuts'

    f = ['p_jaw_f_p ; 730 ; 13_45_58.csv_30.csv']


    for filename in f:
        df = pd.read_csv(f'{cut_imu_path}/{filename}')
        print(df)

