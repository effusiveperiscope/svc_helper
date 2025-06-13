from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import torch
from fairseq import checkpoint_utils
from scipy import signal
import numpy as np
import torch.nn.functional as F
from einops import rearrange
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

from svc_helper.sfeatures.whisper.audio import (
    SAMPLE_RATE as SVC5W_SAMPLE_RATE, pad_or_trim, load_audio,
    log_mel_spectrogram)
from svc_helper.sfeatures.whisper.model import Whisper, ModelDimensions
from svc_helper.sfeatures.svc5.hubert_model import hubert_soft
from svc_helper.sfeatures.svc5.encoder import TextEncoder
from svc_helper.pitch.rmvpe import RMVPEModel
from svc_helper.pitch.utils import f0_to_coarse
from svc_helper.svc.rvc.lib.audio import load_audio

class RVCHubertModel:
    expected_sample_rate = 16000
    def __init__(self, device = torch.device('cpu'), **kwargs):
        rvc_hubert_path = kwargs.get('rvc_hubert_path', hf_hub_download(
            repo_id='therealvul/svc_helper', filename='rvc_hubert.pt'))

        # Monkey patch for fairseq
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            # Ensure weights_only=False unless explicitly set
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [rvc_hubert_path], suffix='')
        torch.load = original_torch_load

        #print('normalize:',saved_cfg.task.normalize)
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

    """ Replicates RVC audio loading and normalization"""
    def load_audio(self, audio_file : str):
        """
        Outputs [1, T, 768]
        """
        audio = load_audio(audio_in, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        return audio

    """ Replicates RVC audio padding - useful for training 
        models that modify extracted features in RVC inference """
    def pad_audio(self, audio : np.ndarray):
        t_pad = self.expected_sample_rate * self.x_pad
        window = self.window_len
        audio = signal.filtfilt(bh, ah, audio)
        #audio = np.pad(audio, (window // 2, window // 2), mode='reflect')
        # RVC overwrites the above line for some reason?
        audio = np.pad(audio, (t_pad, t_pad), mode='reflect')
        return audio

    def extract_features(self, audio: torch.Tensor, **kwargs):
        if type(audio) == np.ndarray:
            audio = torch.from_numpy(audio)
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

class SVC5WhisperModel:
    expected_sample_rate = SVC5W_SAMPLE_RATE
    def __init__(self, device = torch.device('cpu'), **kwargs):
        whisper_path = kwargs.get('whisper_path',
            hf_hub_download(repo_id='therealvul/svc_helper',
                filename='svc5_whisper_large-v2.pt'))

        checkpoint = torch.load(whisper_path, map_location='cpu')
        dims = ModelDimensions(**checkpoint['dims'])
        model = Whisper(dims)
        del model.decoder
        # cut = len(model.encoder.blocks) // 4
        # cut = -1 * cut
        # del model.encoder.blocks[cut:]
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        model.to(device)
        if kwargs.get('is_half'):
            model = model.half()
        self.model = model
        self.device = device
        self.is_half = kwargs.get('is_half', False)

    def extract_features(self, audio: torch.Tensor, **kwargs):
        """
        Outputs [T, 1280]
        """
        feats = audio
        if type(feats) == np.ndarray:
            feats = torch.from_numpy(feats)
        if feats.dim() == 2: # stereo
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        audln = audio.shape[0]
        ppgln = audln // 320
        feats = pad_or_trim(feats)
        # torchaudio only supports 32bit float
        mel = log_mel_spectrogram(feats.float()).to(self.device)
        if self.is_half:
            mel = mel.half()
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu()
            if self.is_half:
                ppg = ppg.half()
            else:
                ppg = ppg.float()
            ppg = ppg[:ppgln,]
        return ppg

class SVC5HubertModel:
    expected_sample_rate = 16000
    def __init__(self, device = torch.device('cpu'), **kwargs):
        hubert_path = kwargs.get('hubert_path',
            hf_hub_download(repo_id='therealvul/svc_helper',
                filename='svc5_hubert-soft.pt'))
        self.model = hubert_soft(hubert_path)
        if kwargs.get('is_half', False):
            self.model = self.model.half()
        self.device = device
        self.is_half = kwargs.get('is_half', False)

    def extract_features(self, audio : torch.Tensor, **kwargs):
        """
        Outputs [T, 256]
        """
        feats = audio
        if type(feats) == np.ndarray:
            feats = torch.from_numpy(feats)
        if self.is_half:
            feats = feats.half()
        feats = feats.to(self.device)
        feats = feats[None, None, :]
        vec = self.model.units(feats)
        return vec.squeeze(0)

class SVC5TextEncoderFullModel:
    expected_sample_rate = 16000
    def __init__(self, device = torch.device('cpu'), **kwargs):
        text_encoder_path = kwargs.get('text_encoder_path',
            hf_hub_download(repo_id='therealvul/svc_helper',
                filename='svc5_text_encoder.pth'))
        self.model = TextEncoder(
            in_channels=1280,
            vec_channels=256,
            out_channels=192,
            hidden_channels=192,
            filter_channels=640,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1
        )
        if kwargs.get('is_half', False):
            self.model = self.model.half()
        self.device = device
        self.is_half = kwargs.get('is_half', False)

        self.whisper = SVC5WhisperModel(device=device, is_half=self.is_half,
            **kwargs)
        self.hubert = SVC5HubertModel(device=device, is_half=self.is_half,
            **kwargs)
        self.rmvpe = RMVPEModel(device=device, is_half=self.is_half,
            **kwargs)

    def extract_features(self, audio : torch.Tensor, **kwargs):
        """
        Outputs [1, 192, T]
        """
        feats = audio
        whisper_feats = self.whisper.extract_features(feats, **kwargs)
        hubert_feats = self.hubert.extract_features(feats, **kwargs)
        pitch = self.rmvpe.extract_pitch(feats, **kwargs)
        coarse_pitch = f0_to_coarse(pitch) # [1, T]
        coarse_pitch = coarse_pitch.to(self.device)

        # Repeat inputs
        whisper_feats = rearrange(
            F.interpolate(
            rearrange(whisper_feats,'t c -> 1 c t'), scale_factor=2.0)
            , '1 c t -> 1 t c')
        hubert_feats = rearrange(
            F.interpolate(
            rearrange(hubert_feats,'t c -> 1 c t'), scale_factor=2.0)
            , '1 c t -> 1 t c')

        min_length = min(whisper_feats.shape[1], hubert_feats.shape[1],
            coarse_pitch.shape[1])

        whisper_feats = whisper_feats[:, :min_length, :]
        hubert_feats = hubert_feats[:, :min_length, :]
        coarse_pitch = coarse_pitch[:, :min_length]

        _, _, _, _, text_encoder_feats = self.model(
            x=whisper_feats, 
            x_lengths=torch.Tensor([whisper_feats.shape[1]])
                .to(self.device).to(torch.long), #.unsqueeze(0),
            v=hubert_feats, 
            f0=coarse_pitch)
        return text_encoder_feats