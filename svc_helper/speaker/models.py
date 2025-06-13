from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch

import re
import json
import fsspec
import numpy as np
import argparse

from argparse import RawTextHelpFormatter
from .svc5.models.lstm import LSTMSpeakerEncoder
from .svc5.config import SpeakerEncoderConfig
from .svc5.utils.audio import AudioProcessor

def read_json(json_path):
    config_dict = {}
    try:
        with fsspec.open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError:
        # backwards compat.
        data = read_json_with_comments(json_path)
    config_dict.update(data)
    return config_dict

def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data

class SVC5SpeakerEncoder(nn.Module):
    def __init__(self, device=torch.device('cpu'), **kwargs):
        super().__init__()
        config_path = kwargs.get('config_path', hf_hub_download(
            repo_id='therealvul/svc_helper', filename='sv5_spk_encoder_config.json'
        ))
        model_path = kwargs.get('model_path', hf_hub_download(
            repo_id='therealvul/svc_helper', filename='svc5_spk_encoder.tar'
        ))

        config_dict = read_json(config_path)

        config = SpeakerEncoderConfig(config_dict)
        config.from_dict(config_dict)

        speaker_encoder = LSTMSpeakerEncoder(
            config.model_params["input_dim"],
            config.model_params["proj_dim"],
            config.model_params["lstm_dim"],
            config.model_params["num_lstm_layers"],
        )
        speaker_encoder.load_checkpoint(model_path, eval=True, 
            use_cuda=(torch.device(device) == torch.device("cuda")))
        self.speaker_encoder = speaker_encoder

        speaker_encoder_ap = AudioProcessor(**config.audio, verbose=False)
        speaker_encoder_ap.do_sound_norm = True
        speaker_encoder_ap.do_trim_silence = True
        self.speaker_encoder_ap = speaker_encoder_ap
        self.device = device

    def extract_feature(self, source_file : str):
        """
        Outputs speaker embedding [256]
        """
        waveform = self.speaker_encoder_ap.load_wav(
            source_file, sr=self.speaker_encoder_ap.sample_rate
        )
        spec = self.speaker_encoder_ap.melspectrogram(waveform)
        spec = torch.from_numpy(spec.T).to(self.device)
        spec = spec.unsqueeze(0)
        embed = self.speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
        embed = embed.squeeze()
        return torch.from_numpy(embed) 