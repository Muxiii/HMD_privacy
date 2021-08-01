import conf
from os import walk
import pandas as pd
import numpy as np
if __name__ == '__main__':

    # import parameters from conf.py
    start_time = conf.START_TIME
    data_freq = conf.DATA_FREQ
    step_length = conf.FRAGMENT_LENGTH
    overlap = conf.OVERLAP
    body = conf.BODY

    imu_path = f'../dataset/{body}/processed_data/imu_data_processed'
    cut_imu_path = f'../dataset/{body}/processed_data/imu_cuts'

    f = []

    for (dirpath, dirnames, filenames) in walk(imu_path):
        f.extend(filenames)
        break

    for filename in f:
        df = pd.read_csv(f'{imu_path}/{filename}')
        s = start_time
        e = start_time + step_length
        i = 1
        print(len(df))
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

    print(i-1)
