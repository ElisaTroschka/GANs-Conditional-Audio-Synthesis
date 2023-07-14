import json
import torch
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import OneHotEncoder
from librosa.feature import melspectrogram
from torch.utils.data import Dataset


class NSynthDataset(Dataset):

    def __init__(self, data_path='data/', stage='train', mel=False, sampling_rate=16000, duration=3, limit=False):
        super(NSynthDataset, self).__init__()
        self.data_path = f'{data_path}nsynth-{stage}'
        self.mel = mel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.limit = limit
        self.json = json.load(open(f'{self.data_path}/examples.json'))
        self._preprocess_dataset()
        self.fnames = self.annot.index.to_list()

    def __len__(self):
        return len(self.fnames) if self.limit == False else self.limit

    def __getitem__(self, i):
        wpath = f'{self.data_path}/audio/{self.fnames[i]}.wav'
        y, sr = librosa.load(wpath, sr=self.sampling_rate, duration=self.duration)
        pitch = torch.tensor(self.annot.loc[self.fnames[i], 'pitch']).unsqueeze(0)
        instr_class = torch.tensor(self.annot.loc[self.fnames[i], 'instrument_class'])
        label = torch.cat((pitch, instr_class))

        if self.mel:
            y = melspectrogram(y=y, sr=sr)

        #self.y_size, self.label_size = y.shape[0], label.shape[0]

        return torch.tensor(y), label

    def _preprocess_dataset(self):
        annot = pd.DataFrame.from_dict(self.json, orient='index')

        # Dropping irrelevant data
        annot.drop(columns=['instrument', 'instrument_str', 'sample_rate'])

        # Removing samples with pitch < 20 or > 110
        to_keep = np.logical_and((20 < annot['pitch']), (annot['pitch'] < 110))
        annot = annot[to_keep]

        # Redefining instrument classes to include both source and family
        annot['instrument_class_str'] = annot['instrument_family_str'] + '_' + annot['instrument_source_str']
        
        # balancing the dataset
        annot = self._balance_data(annot)
        
        # onehot encoding to be used as cond info
        annot['instrument_class'] = torch.tensor(pd.get_dummies(annot['instrument_class_str']).values, dtype=torch.int).tolist()

        self.annot = annot
        self.label_size = 1 + self.annot['instrument_class_str'].nunique()
        self.y_size = self.sampling_rate * self.duration if not self.mel else (128, 94)

    
    def _balance_data(self, annot, min=50, max=51, seed=102):
        val_counts = annot.value_counts('instrument_class_str')
        under_min = val_counts[val_counts < min].index
        over_max = val_counts[val_counts > max].index
        
        # discard classes with value count below min
        annot = annot[~annot['instrument_class_str'].isin(under_min)]
        
        # sample from classes with value count exceeding max
        np.random.seed(seed)  # Set a seed for reproducibility
        for c in over_max:
            cls_indices = annot[annot['instrument_class_str'] == c].index
            tot_to_drop = len(cls_indices) - max
            indices_to_drop = np.random.choice(cls_indices, size=tot_to_drop, replace=False)
            annot = annot.drop(indices_to_drop)
            
        return annot