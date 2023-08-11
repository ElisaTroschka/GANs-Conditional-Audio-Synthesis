import numpy as np
import librosa
import torch


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
    f0, p, v = librosa.pyin(audio.numpy(), sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = f0[np.argmax(p)]
    pitch = hz_to_midi(pitch)
    return pitch


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