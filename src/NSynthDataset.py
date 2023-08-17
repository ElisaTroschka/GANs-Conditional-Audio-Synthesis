import json
import torch
import pandas as pd
import numpy as np
import librosa
import sys
from sklearn.preprocessing import OneHotEncoder
from librosa.feature import melspectrogram
from torch.utils.data import Dataset
from src.utils import midi_to_hz


class NSynthDataset(Dataset):
    """
    
    Args:
    
        data_path (str): Path to folder containing the dataset.
        stage (str): One of 'train', 'test' or 'val'.
        mel (bool): Whether to extract mel-spectrograms.
        sampling_rate (int): Sampling rate in Hz.
        duration (float): Audio duration in seconds.
        min_class_count (int): Minimum number of samples per class to use in class balancing. Only used when stage == 'train'.
        max_class_count (int): Maximum number of samples per class to use in class balancing. Only used when stage == 'train'.
        cond_classes (List(str) or tuple(str)): Classes to keep during class balancing (must match train set classes). Only used when stage == 'valid' or 'test'.
        
    """

    def __init__(self, 
                 data_path='data/', 
                 stage='train', 
                 mel=False, 
                 sampling_rate=8192, 
                 duration=2, 
                 min_class_count=2000, 
                 max_class_count=2500, 
                 cond_classes=None, 
                 pitched_z=False,
                 z_size=1000
                ):
        
        super(NSynthDataset, self).__init__()
        self.stage = stage
        self.data_path = f'{data_path}nsynth-{stage}'
        self.mel = mel
        self.sampling_rate = sampling_rate
        self.duration = duration

        if stage not in ('train', 'test', 'valid'):
            raise ValueError('stage must be one of \'train\', \'test\', \'valid\'')
        if stage != 'train' and cond_classes is None:
            raise ValueError('incompatible arguments: \'test\' or \'valid\' stages require cond_classes to be specified')

        self.min_class_count = min_class_count
        self.max_class_count = max_class_count
        self.cond_classes = cond_classes
        self.pitched_z = pitched_z
        self.z_size = z_size

        self.json = json.load(open(f'{self.data_path}/examples.json'))
        self._preprocess_dataset()
        self.fnames = self.annot.index.to_list()
            

    def __len__(self):
        return len(self.fnames)

    
    def __getitem__(self, i):
        # Loading audio file and normalizing
        wpath = f'{self.data_path}/audio/{self.fnames[i]}.wav'
        y, sr = librosa.load(wpath, sr=self.sampling_rate, duration=self.duration)
        
        # Constructing mel spec
        if self.mel:
            y = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=128, n_mels=128)
            
        # Normalize each frequency bin to have zero mean and unit variance
        y = (y - np.mean(y)) / np.std(y)
        # Clip to 3 standard deviations
        y = np.clip(y, -3, 3)
        # Rescale to [-1, 1]
        y = (y / 3).clip(-1, 1)

        
        # constructing label
        instr_class = torch.tensor(self.annot.loc[self.fnames[i], 'instrument_class'])
        
        if self.pitched_z:
            label = instr_class
            z = librosa.tone(midi_to_hz(self.get_pitch(i)), sr=self.sampling_rate, length=self.z_size).type(torch.float32)
        else:
            pitch = torch.tensor(self.get_pitch(i))#.unsqueeze(0)
            label = torch.cat((pitch, instr_class))
            z = 2 * torch.rand(self.z_size) - 1
            
        return torch.tensor(y), label, z
    

    def _preprocess_dataset(self):
        annot = pd.DataFrame.from_dict(self.json, orient='index')

        # Dropping irrelevant data
        annot.drop(columns=['instrument', 'instrument_str', 'sample_rate'])

        # Removing samples with pitch < 21 or > 108
        #to_keep = np.logical_and((21 < annot['pitch']), (annot['pitch'] < 108))
        to_keep = np.logical_and((40 < annot['pitch']), (annot['pitch'] < 80))
        annot = annot[to_keep]

        # Redefining instrument classes to include both source and family
        annot['instrument_class_str'] = annot['instrument_family_str'] + '_' + annot['instrument_source_str']
        
        # balancing the dataset
        annot = self._balance_data(annot)
        
        # onehot encoding to be used as cond info
        if self.stage == 'train':
            self.cond_classes = np.array(sorted(annot['instrument_class_str'].unique())).reshape(1, -1).tolist()
            
        enc = OneHotEncoder(categories=self.cond_classes)
        encoded_features = enc.fit_transform(annot[['instrument_class_str']]).toarray().astype(int).tolist()
        encoded_df = pd.DataFrame({'instrument_class':encoded_features}, index=annot.index)
        annot = pd.concat((annot, encoded_df), axis=1)

        self.annot = annot
        self.label_size = self.annot['instrument_class_str'].nunique()
        if not self.pitched_z:
            self.label_size += 1
        self.y_size = self.sampling_rate * self.duration if not self.mel else 128

    
    def _balance_data(self, annot, seed=102):
        """
        Balances instrument_class so that each class has min_class_count < count < max_class_count
        """
        if self.stage == 'train':

            min, max = self.min_class_count, self.max_class_count
            val_counts = annot.value_counts('instrument_class_str')
            under_min = val_counts[val_counts < min].index
            over_max = val_counts[val_counts > max].index

            # discard classes with value count below min
            annot = annot[~annot['instrument_class_str'].isin(under_min)]

            # sample from classes with value count exceeding max
            np.random.seed(seed)
            for c in over_max:
                if c not in under_min:
                    cls_indices = annot[annot['instrument_class_str'] == c].index
                    tot_to_drop = len(cls_indices) - max
                    indices_to_drop = np.random.choice(cls_indices, size=tot_to_drop, replace=False)
                    annot = annot.drop(indices_to_drop)
        else:
            annot = annot[annot['instrument_class_str'].isin(self.cond_classes[0])]
            
        return annot
    
    
    def get_pitch(self, i):
        return torch.tensor(self.annot.loc[self.fnames[i], 'pitch']).unsqueeze(0)
