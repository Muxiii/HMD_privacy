
"""
    file path information
"""

BODY = 'jaw_f_p'

"""
    audio file parameters
"""

DATA_FREQ = 1000
SECONDS = 50
FRAGMENT_LENGTH = 0.5
OVERLAP = 0.25

"""
    imu data parameters
"""
#   define the start point, before which the redundant data would be abandoned
#   p.s.   this is to avoid big fluctuations (in the first 0.X secs) caused by high-pass filtering
#   p.p.s  log time time of imu would be prolonged accordingly, so does the audio record (which would shift backwards)
START_TIME = 0.1
HIGH_PASS = 20
