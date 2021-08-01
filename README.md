#### Yuchen Memo

##### conf.py 

​	set essential parameters:

​	RECORD_SECONDS, DATA_FREQ, ...

​	imported as a package

##### log.py 

​	log imu data and audio simultaneously

​	input : data from Arduino serial

​	 output: both raw and processed IMU data, and audio record

##### speech_reconstruction.py

​	input : IMU data (processed)

​	output : reconstructed speech and reduced version

##### cut_up.py 

​	cut files into fragments for training

​	set: step length(0.5s for now), overlap(0 for now)

​	input : IMU data (processed), audio, reconstructed audio

​	output : IMU cuts, audio cuts, reconstruced audio cuts

​	

