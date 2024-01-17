# Conditional Audio Synthesis with GANs

This project focuses on deep audio generation. The main objective is to train a deep learning model to generate musical notes of a desired pitch and instrument, obtaining a tool that can boost creativity in music composition. To do so, this work explores two different approaches, both based on generative adversarial networks: the first one is to operate directly on raw audio, while the second one is to use intermediate image representations, by projecting the audio into Mel-spectrograms. The results of the two are evaluated through various metrics and their results are compared, showing that both are fit for the generation of short sounds and of musical notes in particular. This project reviews both approaches highlighting the strengths and shortcomings of each, ultimately setting a baseline to work on and improve.

NSynth: https://magenta.tensorflow.org/datasets/nsynth#files
