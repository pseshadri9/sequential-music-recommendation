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


if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    
    print("Name of the current run (press ENTER for default):")
    exp_name = input()
    #torch.manual_seed(config['exp_params']['manual_seed'])
    seed_everything(config['exp_params']['manual_seed'])
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    start_time = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S")
    print(f'Run started at {start_time} with args:\n\n{config}\n')

    #seed_everything(config['exp_params']['manual_seed'], True)

    print('Loading data....')
    data = SpotifyDataModule(config['data_params']['data_path'], config['data_params']['batch_size'])

    model = VanillaTransformer(**config['model_params'], vocab_size=len(data.vocab))

    #model = torch.compile(model)

    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'] + f': {exp_name}',)
    
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=1, 
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
    if config['data_params']['load_ckpt']:
        runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=1,
                                     dirpath =config['data_params']['ckpt_path'], 
                                     monitor= "val_loss",
                                     save_last= True,
                                     every_n_epochs=1),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 limit_train_batches=0, limit_val_batches=0,
                 **config['trainer_params'])
        model = VanillaTransformer.load_from_checkpoint(config['data_params']['ckpt_type'])
        model.eval()
        runner.fit(model, data.train_dataloader(), data.val_dataloader())
        runner.test(ckpt_path = 'best', dataloaders = data.test_dataloader())
    
    else:
        try:
            runner.fit(model, data.train_dataloader(), data.val_dataloader())
            #train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader()) #,
        except KeyboardInterrupt:
            pass

        runner.test(ckpt_path="best", dataloaders = data.test_dataloader())
