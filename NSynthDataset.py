import json
import torch
import librosa
from torch.utils.data import Dataset

class NSynthDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path='data/', stage='train', sampling_rate=16000, duration=3, labeling=['pitch', 'instrument_family']):
        super(NSynthDataset, self).__init__()
        self.data_path = f'{data_path}nsynth-{stage}'
        self.labeling = labeling
        self.sampling_rate = None
        self.duration = 1
        self.annot = json.load(open(f'{self.data_path}/examples.json'))
        self.fnames = list(self.annot.keys())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        wpath = f'{self.data_path}/audio/{self.fnames[i]}.wav'
        wav = torch.tensor(librosa.load(wpath, sr=self.sampling_rate, duration=self.duration)[0])
        label = torch.tensor([self.annot[self.fnames[i]][feature] for feature in self.labeling])
        return wav, label
        