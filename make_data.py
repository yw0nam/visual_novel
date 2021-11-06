import os, subprocess, json
from glob import glob



command = ("ffmpeg -i {input_path} -b 768k -ac 1 -ar 48000 {output_path}.wav".format(input_path=path, output_path=os.path.join(save_path, os.path.basename(path).split('.')[0])))
output = subprocess.call(command, shell=True, stdout=None)
