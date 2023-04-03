from utility import run
import argparse
import datetime
from data_process import SpotifyDataModule
from models import VanillaTransformer
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
'''
Define experiment run arguments here
'''
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logdir', type=str, default='logger_runs/', help='directory to save logs')
    parser.add_argument('--datadir', type=str, default='/mnt/data2/pavans/pavans/training_set/', help='directory to save data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--config', type=str, default='config.yml')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args.start_time = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S")
    print(f'Run started at {args.start_time} with args:\n\n{args}')

    seed_everything(config['exp_params']['manual_seed'], True)

    data = SpotifyDataModule(args.datadir, args.batch_size)

    model = VanillaTransformer(**config['model_params'])

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)
    
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])
    
    runner.fit(model, datamodule=data)
