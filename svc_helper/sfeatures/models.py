from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from fairseq import checkpoint_utils
import numpy as np
import torch

class RVCHubertModel:
    expected_sample_rate = 16000
    def __init__(self, device = torch.device('cpu'), **kwargs):
        rvc_hubert_path = hf_hub_download(
            repo_id='therealvul/svc_helper', filename='rvc_hubert.pt')

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [rvc_hubert_path], suffix='')
        model = models[0]
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        if kwargs.get('is_half'):
            model = model.half()
        else:
            model = model.float()
        self.model = model.eval()
        self.device = device

    def extract_features(self, audio: torch.Tensor, **kwargs):
        feats = audio
        if kwargs.get('is_half'):
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