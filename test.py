from transformers import WhisperProcessor, WhisperForConditionalGeneration
from svc_helper.svc.rvc import RVCModel
from svc_helper.sfeatures.models import RVCHubertModel
from huggingface_hub import hf_hub_download
from svc_helper.svc.rvc.lib.audio import load_audio, wav2
import torch

audio_in = './srcaudio5.flac'
rvc_model = RVCModel()
test_model_path = hf_hub_download(repo_id='therealvul/RVCv2', 
    filename='RarityS1/Rarity.pth')
test_index_path = hf_hub_download(repo_id='therealvul/RVCv2', 
    filename='RarityS1/added_IVF1866_Flat_nprobe_1_Rarity_v2.index')
rvc_model.load_model(model_path = test_model_path,
    index_path = test_index_path)


wav_opt = rvc_model.infer_file(audio_in, index_rate=0.0, transpose=0)

hubert_model = RVCHubertModel(is_half=True)
import librosa
import numpy as np

audio = load_audio(audio_in, 16000)
audio_max = np.abs(audio).max() / 0.95
if audio_max > 1:
    audio /= audio_max
print(len(audio))
print(audio[0])
print(audio[-1])

padded_audio = hubert_model.pad_audio(audio)
print(len(padded_audio))
#import pdb
#pdb.set_trace()

import soundfile as sf
sf.write('audiofromus.wav', data=padded_audio, samplerate=16000)