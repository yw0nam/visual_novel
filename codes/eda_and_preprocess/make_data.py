import os, subprocess, argparse
from tqdm import tqdm
from glob import glob
def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', required=True)
    p.add_argument('--voice_path', required=True)
    config = p.parse_args()
    return config

def main(config):
    if os.path.exists(config.save_path):
        pass
    else:
        os.mkdir(config.save_path)
    voice_pathes = glob(config.voice_path+'/*.ogg')
    for path in tqdm(voice_pathes):
        command = ("ffmpeg -hide_banner -loglevel error -i {input_path} -b 768k -ac 1 -ar 22050 {output_path}.wav".format(input_path=path, output_path=os.path.join(config.save_path, os.path.basename(path).split('.')[0])))
        output = subprocess.call(command, shell=True, stdout=None)
            
if __name__ == '__main__':
    config = define_argparser()
    main(config)
