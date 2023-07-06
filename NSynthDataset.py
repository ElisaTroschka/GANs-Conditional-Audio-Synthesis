import json
import torch
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from librosa.feature import melspectrogram
from torch.utils.data import Dataset


class NSynthDataset(Dataset):

    def __init__(self, data_path='data/', stage='train', mel=False, sampling_rate=16000, duration=3):
        super(NSynthDataset, self).__init__()
        self.data_path = f'{data_path}nsynth-{stage}'
        self.mel = mel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.json = json.load(open(f'{self.data_path}/examples.json'))
        self._preprocess_dataset()
        self.fnames = list(self.json.keys())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        wpath = f'{self.data_path}/audio/{self.fnames[i]}.wav'
        y, sr = librosa.load(wpath, sr=self.sampling_rate, duration=self.duration)
        label = torch.tensor(self.annot.iloc[self.fnames[i], 'instrument_class'])

        if self.mel:
            y = melspectrogram(y=y, sr=sr)

        self.y_size, self.label_size = y.shape[0], label.shape[0]

        return torch.tensor(y), label

    def _preprocess_dataset(self):
        annot = pd.DataFrame(self.json)

        # Dropping irrelevant data
        annot.drop(['instrument', 'instrument_str', 'sample_rate'])

        # Removing samples with pitch < 20 or > 110
        annot = annot[20 < annot['pitch'] < 110]

        # Redefining instrument classes
        annot['instrument_class_str'] = annot['instrument_family_str'] + '_' + annot['instrument_source_str']
        annot['instrument_class'] = LabelEncoder().fit_transform(annot['instrument_class_str'])

        print(annot)
        self.annot = annot

