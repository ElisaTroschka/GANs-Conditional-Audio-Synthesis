import numpy as np
import librosa
from librosa.feature.inverse import mel_to_audio
from librosa import hz_to_mel
from IPython.display import Audio
import torch
import matplotlib.pyplot as plt
import scipy
from scipy.stats import mode


def hz_to_midi(f):
    """
    Converts a frequence in Hz to the correspondent MIDI note.
    """
    midi_ref_freq = 440.0
    midi_ref_note = 69
    return 12 * np.log2(f / midi_ref_freq) + midi_ref_note


def midi_to_hz(m):
    """
    Converts a MIDI note to the correspondent frequence in Hz.
    """
    midi_ref_freq = 440.0
    midi_ref_note = 69
    return 2 ** ((m - midi_ref_note) / 12) * midi_ref_freq


def estimate_pitch(audio, sr):
    """
    Applies pYIN pitch estimation to approximately compute the predominant pitch.
    """
    pitch = librosa.yin(np.array(audio), sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=64)
    #pitch = f0[np.argmax(p)]
    #pitch = mode(f0, keepdims=False)[0]
    #pitch=np.mean(f0)
    pitch = np.array([hz_to_midi(p) for p in pitch])
    return np.median(pitch), pitch#mode(pitch, keepdims=False)[0], pitch


def flip_random_elements(target_r, flip_prob, device):
    """
    Flips elements of a tensor of booleans with probability `flip_prob`
    """
    # Generate a tensor of random probabilities for each element in target_r
    rand_probs = torch.rand(target_r.shape).to(device)

    # Create a mask where elements with random probabilities less than flip_prob are set to 0
    mask = (rand_probs > flip_prob).float()
    target_r_flipped = target_r * mask

    return target_r_flipped

def display_audio_sample(i, train_set, G):
    w, l, z = train_set.__getitem__(i)
    G.eval()
    s = G.forward(z.unsqueeze(0).to(torch.device('cuda')), l.unsqueeze(0).to(torch.device('cuda')))
    s.to(torch.device('cpu'))
    s = s.detach().cpu()
    if train_set.mel:
        s = mel_to_audio(np.array(s), sr=train_set.sampling_rate, n_fft=1024, hop_length=128)
    return Audio(s, rate=train_set.sampling_rate)
    
    
def display_mel_sample(i, train_set, G, db=False):
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
    
    
def clean(y, sr, f=np.array([2680, 2780])):
    
    n_fft = 1024
    hop_length = 128
    noisy_stft = librosa.stft(np.array(y), n_fft=n_fft, hop_length=hop_length)
    cutoff_freq = f  # Adjust this cutoff frequency as needed
    nyquist_freq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist_freq
    normal_cutoff = np.clip(normal_cutoff, 0.0, 0.99)
    b, a = scipy.signal.butter(6, normal_cutoff, btype='bandstop', analog=False)
    clean_audio = scipy.signal.lfilter(b, a, y)
    return clean_audio


def filter_mel(mel, fmin=2680, fmax=2780):
    mel_freq_min = librosa.hz_to_mel(fmin)
    mel_freq_max = librosa.hz_to_mel(fmax)

    # Set values in the mel spectrogram to zero for the specified frequency range
    mel[(mel_freq_min <= mel) & (mel <= mel_freq_max)] = mel.min()
    return mel