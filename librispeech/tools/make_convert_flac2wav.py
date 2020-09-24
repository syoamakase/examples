import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wavlist')
    parser.add_argument('--save_dir')
    parser.add_argument('-n', '--num_subdir', type=int)
    args = parser.parse_args()
    wavlist = args.wavlist
    save_dir = args.save_dir
    sampling_rate = args.sampling_rate
    num_subdir = args.num_subdir

    with open(wavlist) as f:
        for wf in f:
            wf = wf.strip()
            basename = os.path.basename(wf)
            save_name = wf.split('/')[num_subdir:]
            save_name = '/'.join(save_name)
            save_path = os.path.join(save_dir, save_name)
            save_path = os.path.abspath(save_path)
            # very high option
            print(f'ffmpeg -i {wf} {save_path}')
