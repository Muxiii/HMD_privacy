import os
from pydub import AudioSegment
import numpy as np

body = 'tap'
"""开始批处理"""

file_path1 = f'../dataset/{body}/processed_data/speech_reconstruction/'    #输入路径
file_path2 = f'../dataset/{body}/processed_data/audio_cuts'   #输出路径
file_path3 = f'../dataset/{body}/processed_data/reconstructed_audio_cuts'   #输出路径

for file in os.listdir(file_path1):              #遍历文件
    path1 = file_path1+'\\'+file
    filename = file.split('.')[0]                #不带 .wav的文件名

"""处理音频文件"""

    audio = AudioSegment.from_file(path1, "wav")
    audio_time = len(audio)                      # 获取待切割音频的时长，单位是毫秒
    cut_parameters = np.arange(1, audio_time / 1000, 10)    # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）
    start_time = int(0)                          # 开始时间设为0

"""根据数组切割音频"""

    for t in cut_parameters:
        stop_time = int(t * 1000)  # pydub以毫秒为单位工作
        # print(stop_time)
        audio_chunk = audio[start_time:stop_time]  # 音频切割按开始时间到结束时间切割
        print("split at [{}:{}] ms".format(start_time, stop_time))
        audio_chunk.export(file_path2 + '\\' + filename + "-{}.wav".format(int(t / 1)), format="wav")  #保存音频文件
        start_time = stop_time - 4000  # 开始时间变为结束时间前4s---------也就是叠加上一段音频末尾的4s
        print('finish')

