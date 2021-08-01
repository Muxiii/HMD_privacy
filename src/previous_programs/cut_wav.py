from scipy.io import wavfile

like = wavfile.read('test.wav')
# print (like)
# 音频结果将返回一个tuple。第一维参数是采样频率，单位为秒；第二维数据是一个ndarray表示歌曲，如果第二维的ndarray只有一个数据表示单声道，两个数据表示立体声。所以，通过控制第二维数据就能对歌曲进行裁剪。
# 对like这个元组第二维数据进行裁剪，所以是like[1];第二维数据中是对音乐数据切分。 start_s表示你想裁剪音频的起始时间；同理end_s表示你裁剪音频的结束时间。乘44100 是因为每秒需要进行44100次采样
# 这里表示对该音频的13-48秒进行截取
wavfile.write('test.wav', 1000, like[1][5*1000:10*1000])

