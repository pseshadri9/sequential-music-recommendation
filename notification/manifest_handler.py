import json
import uuid 
import os
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np

FILE_EXT = '.json'

#manifest fields
CONFIG = 'config'
EVAL = 'eval'
METADATA = 'metadata'
VIZ_DIR = 'viz'
MODEL_PATH = 'model_path'

class manifestHandler():
    def __init__(self, config = dict(), eval = dict(), metadata = dict(), model_path='', name = str(uuid.uuid4()), save_path='logger_runs/manifest/'):
        self.save_path = save_path
        self.name = name 
        self.config = config 
        self.eval = eval 
        self.metadata = metadata
        self.model_path = model_path
        self.manifest = {CONFIG: config, EVAL: eval, METADATA: metadata, MODEL_PATH:model_path}
        self.viz_filepath = None
    
    def add(self, X: dict[str, dict]):
        for k in X.keys():
            if k not in self.manifest.keys():
                self.manifest.update(X)
            else:
                self.manifest[k].update(X[k])
    
    def save(self):
        '''
        saves json file of training/evaluation manifest with following entries:

            CONFIG: specified data/model configuration for given run
            EVAL: summary of evaluation metrics for given run
            METADATA: [partially deprecated], any relevant metadata for given run
            MODEL_PATH: filepath to best performing checkpoint for given run

        File name: [encoder_type]-r-[training radius]-tt-[training threshold]-[count].json

        '''
        assert not self.save_path is None

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        dataset_name = self.manifest[CONFIG]['logging_params']['name']

        self.save_path = os.path.join(self.save_path, dataset_name)

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        dir_contents = os.listdir(self.save_path)

        count = len([x for x in dir_contents if x.startswith(self.name)])
        self.name = f'{self.name}-{count}'

        #os.mkdir(os.path.join(self.save_path, self.name))

        self.manifest = self.proc_dict(self.manifest)
        print(self.manifest)
        #self.create_and_save_graphs()

        with open(os.path.join(self.save_path, f'{self.name}{FILE_EXT}'), 'w') as f:
            json.dump(self.manifest, f, indent=4)
    
    '''
    if any key in manifest dict is a torch tensor, convert to string
    '''
    def proc_dict(self, d):
        for x in d.keys():
            if type(d[x]) == dict:
                if type(x) == Tensor:
                    d[str(x.item())] = self.proc_dict(d[x])
                else:
                    d[str(x)] = self.proc_dict(d[x])

            elif type(d[x]) == Tensor:
                if type(x) == Tensor:
                    d[str(x.item())] = str(d[x].item())
                else:
                    d[str(x)] = str(d[x].item())
            else:
                if type(x) == Tensor:
                    d[str(x.item())] = str(d[x])
                else:
                    d[str(x)] = str(d[x])
        return d

    def create_and_save_graphs(self):
        if len(self.manifest[EVAL]) == 0:
            raise Exception('No logged evaluation metrics')
        
        self.viz_filepath = os.path.join(self.save_path, self.name, VIZ_DIR)

        if not os.path.isdir(self.viz_filepath):
            os.mkdir(self.viz_filepath)
        for z in ['', 'window ']:
            for x in ['Precision', 'Recall', 'F1-Score', 'Support']:
                neg_hit = [round(float(self.manifest[EVAL][y][f'{z}0.0'][x.lower()]), 2) for y in self.manifest[EVAL].keys()]
                pos_hit = [round(float(self.manifest[EVAL][y][f'{z}1.0'][x.lower()]), 2) for y in self.manifest[EVAL].keys()]
                print(neg_hit, pos_hit)

                self.make_graph(neg_hit, pos_hit,
                                boundary=self.manifest[CONFIG]['dataloader_params']['boundary'],
                                Threshold=self.manifest[CONFIG]['dataloader_params']['train_threshold'], 
                                metric=x, prefix=z)

    def make_graph(self, neg_hit, pos_hit, boundary=1, Threshold=1, metric='recall', EXT= '.png', prefix=''):

        labels = [str(x) for x in range(1, len(neg_hit) + 1)]
        title = f'{metric} @ Threshold=K \nfor Boundary = {boundary}, Training Threshold={Threshold}'
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars


        fig, ax = plt.subplots()
        rects1 = ax.bar(x + width/2, neg_hit, width, label='Negative Hit')
        rects3 = ax.bar(x - width/2, pos_hit, width, label='Positive Hit')
        #rects2 = ax.bar(x + width/2, women_means, width, label='Clarinet solo')

        """Add some text for labels, title and custom x-axis tick labels, etc."""
        ax.set_ylabel(metric)
        ax.set_xlabel('Pedestrian Threshold')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='center right')


        def autolabel(rects):
            """Attach a text label above each bar in rects, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects3)

        fig.tight_layout()

        plt.savefig(os.path.join(self.viz_filepath, f'{prefix}{metric}{EXT}'), bbox_inches='tight')

        