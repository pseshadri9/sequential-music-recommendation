import datetime
from data_process import SpotifyDataModule
from models import VanillaTransformer
import yaml
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary


def train(runner, model, dual_train=False, config=None):
    if dual_train:
        print('TRAINING EMBEDDINGS')
        model.return_skip = False
        #runner.init_optimizers(model)
        runner.fit(model, data.train_dataloader(), data.val_dataloader())

        model = VanillaTransformer.load_from_checkpoint(runner.checkpoint_callback.best_model_path)

        model.return_skip = True
        print('FINE TUNING NEGATIVE SAMPLES')
        model.vocab.weight.requires_grad = False
        #model.decoder_bias.requires_grad = False
        #runner.fit_loop.max_epochs= config['trainer_params']['max_epochs'] * 2
        runner, _ , _ = get_trainer(config)
        runner.fit(model, data.train_dataloader(), data.val_dataloader())

    else:
        print("TRAINING FULL PASS")
        runner.fit(model, data.train_dataloader(), data.val_dataloader())
    
    model.return_skip = False
    return runner, model

def get_trainer(config, ckpt_path = None):
    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'] + f': {exp_name}',)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True,
                                     every_n_epochs=1)
    
    runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     checkpoint_callback,
                 ],
                 gradient_clip_val=5,
                 #strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])
    return runner, tb_logger, checkpoint_callback

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    
    print("Name of the current run (press ENTER for default):")
    if config['dev']:
        exp_name = 'dev'
        #config['trainer_params']['accelerator'] = 'cpu'
        #del config['trainer_params']['devices']
    else:
        exp_name = input()
    #torch.manual_seed(config['exp_params']['manual_seed'])
    seed_everything(config['exp_params']['manual_seed'])
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    start_time, start_datetime = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S"), datetime.datetime.now()
    print(f'Run started at {start_time} with args:\n\n{config}\n')

    #seed_everything(config['exp_params']['manual_seed'], True)

    print('Loading data....')
    data = SpotifyDataModule(config['data_params']['data_path'], config['data_params']['batch_size'], dev=config['dev'])

    model = VanillaTransformer(**config['model_params'], vocab_size=len(data.vocab))

    #model = torch.compile(model)

    runner, tb_logger, checkpoint_callback = get_trainer(config)
    
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
                    EarlyStopping(monitor='val_loss')
                 ],
                 #strategy=DDPStrategy(find_unused_parameters=True),
                 limit_train_batches=0, limit_val_batches=0,
                 **config['trainer_params'])
        model = VanillaTransformer.load_from_checkpoint(config['data_params']['ckpt_type'])
        model.eval()
        runner.fit(model, data.train_dataloader(), data.val_dataloader())
        runner.test(ckpt_path = 'best', dataloaders = data.test_dataloader())
    
    else:
        try:
            runner, model = train(runner, model, dual_train=config['dual_train'], config=config)
            #runner.fit(model, data.train_dataloader(), data.val_dataloader())
            #train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader()) #,
        except KeyboardInterrupt:
            pass

        runner.test(ckpt_path="best", dataloaders = data.test_dataloader())
    
    print('TIME ELAPSED: ',round((datetime.datetime.now() - start_datetime).total_seconds() / 3600, 2), 'HOURS')
