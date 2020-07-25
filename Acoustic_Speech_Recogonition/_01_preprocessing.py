
# data preprocessing and feature extraction

# Dataset: LibriSpeech for English - read speech, designed for training and evaluating models for ASR
# Features: 1. Spectrogram   2. MFCCs

from data_generator import vis_train_features
from IPython.display import Markdown, display
from IPython.display import audio
from data_generator import vis_train_features, plot_raw_audio, plot_spectrogram_feature, plot_mfcc_feature

# extract label and audio feature for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()

plot_raw_audio(vis_raw_audio)
display(Markdown('** Shape of Audio Signal**:' + str(vis_raw_audio.shape)))
display(Markdow('**Transcript**:' + str(vis_text)))

# Step 1: Acoustic Feature for Speech Recognition

# Feature 1: Spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)

# Feature 2: MFCC s
plot_mfcc_feature(vis_mfcc_feature)


