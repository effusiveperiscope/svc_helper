from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from fairseq import checkpoint_utils
from scipy import signal
import numpy as np
import torch
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class RVCHubertModel:
    expected_sample_rate = 16000
    def __init__(self, device = torch.device('cpu'), **kwargs):
        rvc_hubert_path = hf_hub_download(
            repo_id='therealvul/svc_helper', filename='rvc_hubert.pt')

        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [rvc_hubert_path], suffix='')
        print('normalize:',saved_cfg.task.normalize)
        model = models[0]
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        if kwargs.get('is_half'):
            model = model.half()
            self.is_half = True
        else:
            model = model.float()
            self.is_half = False

        self.window_len = kwargs.get('window_len', 160)
        self.x_pad = 3 if self.is_half else 1
        self.model = model.eval()
        self.device = device

    """ Replicates RVC audio padding - useful for training 
        models that modify extracted features in RVC inference """
    def pad_audio(self, audio : np.ndarray):
        t_pad = self.expected_sample_rate * self.x_pad
        window = self.window_len
        audio = signal.filtfilt(bh, ah, audio)
        audio = np.pad(audio, (window // 2, window // 2), mode='reflect')
        audio = np.pad(audio, (t_pad, t_pad), mode='reflect')
        return audio

    def extract_features(self, audio: torch.Tensor, **kwargs):
        feats = audio
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2: # stereo
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        version = kwargs.get('version')
        inputs = {
            'source': feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0]) if version == "v1" else logits[0]

        return feats