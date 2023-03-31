from utility import run
import argparse
import datetime
from data_process import SpotifyDataModule

'''
Define experiment run arguments here
'''
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logdir', type=str, default='logger_runs/', help='directory to save logs')
    parser.add_argument('--datadir', type=str, default='/mnt/data2/pavans/pavans/training_set/', help='directory to save data')
    parser.add_argument('--batch_size', type=int, default=256)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.start_time = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S")
    print(f'Run started at {args.start_time} with args:\n\n{args}')
    
    x = SpotifyDataModule(args.datadir, args.batch_size)
