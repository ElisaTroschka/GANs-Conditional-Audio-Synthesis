import json
import torch
import librosa
from librosa.feature import melspectrogram
from torch.utils.data import Dataset


class NSynthDataset(Dataset):

    def __init__(self, data_path='data/', stage='train', mel=False, sampling_rate=16000, duration=3, labeling=('pitch', 'instrument_family')):
        super(NSynthDataset, self).__init__()
        self.data_path = f'{data_path}nsynth-{stage}'
        self.labeling = labeling
        self.mel = mel
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.annot = json.load(open(f'{self.data_path}/examples.json'))
        self.fnames = list(self.annot.keys())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        wpath = f'{self.data_path}/audio/{self.fnames[i]}.wav'
        y, sr = librosa.load(wpath, sr=self.sampling_rate, duration=self.duration)
        label = torch.tensor([self.annot[self.fnames[i]][feature] for feature in self.labeling])

        if self.mel:
            y = melspectrogram(y=y, sr=sr)

        return torch.tensor(y), label
        