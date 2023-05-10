from utility import run
import argparse
import datetime
from data_process import SpotifyDataModule
from models import VanillaTransformer
import yaml
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary
'''
Define experiment run arguments here
'''
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logdir', type=str, default='logger_runs/', help='directory to save logs')
    parser.add_argument('--datadir', type=str, default='/mnt/data2/pavans/pavans/training_set/', help='directory to save data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--config', type=str, default='config.yml')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    
    print("Name of the current run (press ENTER for default):")
    exp_name = input()
    #torch.manual_seed(config['exp_params']['manual_seed'])
    seed_everything(config['exp_params']['manual_seed'])
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    args.start_time = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S")
    print(f'Run started at {args.start_time} with args:\n\n{config}\n')

    #seed_everything(config['exp_params']['manual_seed'], True)

    print('Loading data....')
    data = SpotifyDataModule(config['data_params']['data_path'], config['data_params']['batch_size'])

    model = VanillaTransformer(**config['model_params'], vocab_size=len(data.vocab))

    #model = torch.compile(model)

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'] + exp_name,)
    
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True,
                                     every_n_epochs=1),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])
    
    print(ModelSummary(model))

    #output = runner.predict(model, data.val_dataloader())

    #topk_orig, total = model.test_top_k(output)

    #print({f'top-{k_i} HR': v / total for k_i, v in top_k.items()})
    if config['data_params']['ckpt_path'] != '':
        runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =config['data_params']['ckpt_path'], 
                                     monitor= "val_loss",
                                     save_last= True,
                                     every_n_epochs=1),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])
        runner.test(model = model, ckpt_path=config['data_params']['ckpt_type'], dataloaders = data.test_dataloader())
    
    else:
        try:
            runner.fit(model, data.train_dataloader(), data.val_dataloader())
            #train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader()) #,
        except KeyboardInterrupt:
            pass

        runner.test(ckpt_path="best", dataloaders = data.test_dataloader())
