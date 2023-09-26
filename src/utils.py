import numpy as np
import librosa
from librosa.feature.inverse import mel_to_audio
from IPython.display import Audio
import torch
import matplotlib.pyplot as plt

def hz_to_midi(f):
    """
    Converts a frequency in Hz to the correspondent MIDI note.
    """
    midi_ref_freq = 440.0
    midi_ref_note = 69
    return 12 * np.log2(f / midi_ref_freq) + midi_ref_note


def midi_to_hz(m):
    """
    Converts a MIDI note to the correspondent frequency in Hz.
    """
    midi_ref_freq = 440.0
    midi_ref_note = 69
    return 2 ** ((m - midi_ref_note) / 12) * midi_ref_freq


def estimate_pitch(audio, sr):
    """
    Computes the predominant pitch by applying YIN frequency estimator.
    """
    pitch = librosa.yin(np.array(audio), sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=64)
    pitch = np.array([hz_to_midi(p) for p in pitch])
    return np.median(pitch), pitch


def flip_random_elements(target_r, flip_prob, device):
    """
    Flips elements of a tensor of booleans with probability `flip_prob`
    """
    rand_probs = torch.rand(target_r.shape).to(device)
    mask = (rand_probs > flip_prob).float()
    target_r_flipped = target_r * mask

    return target_r_flipped


def display_audio_sample(i, train_set, G):
    """
    Displays the output of the generator G given the label of i-th element of dataset as input
    :param i: index of item in train_set
    :param train_set: instance of NSynthDataset
    :param G: instance of WaveGANGenerator
    :return: audio output display
    """
    w, l, z = train_set.__getitem__(i)
    G.eval()
    s = G.forward(z.unsqueeze(0).to(torch.device('cuda')), l.unsqueeze(0).to(torch.device('cuda')))
    s.to(torch.device('cpu'))
    s = s.detach().cpu()
    if train_set.mel:
        s = mel_to_audio(np.array(s), sr=train_set.sampling_rate, n_fft=1024, hop_length=128)
    return Audio(s, rate=train_set.sampling_rate)
    
    
def display_mel_sample(i, train_set, G, db=False):
    """
    Displays the output of the generator G given the label of i-th element of dataset as input
    :param db: Whether the Mel-spectrogram has already been converted to decibels
    :param i: index of item in train_set
    :param train_set: instance of NSynthDataset
    :param G: instance of SpecGANGenerator
    :return: Mel-spectrogram output display
    """
    w, l, z = train_set.__getitem__(i)
    G.eval()
    s = G.forward(z.unsqueeze(0).to(torch.device('cuda')), l.unsqueeze(0).to(torch.device('cuda')))
    s = s.detach().cpu()
    
    plt.figure(figsize=(5, 3))
    if db:
        librosa.display.specshow(np.array(s), y_axis='mel', x_axis='time', cmap='magma')
    else:
        librosa.display.specshow(librosa.power_to_db(s, ref=np.max),  y_axis='mel', x_axis='time', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
