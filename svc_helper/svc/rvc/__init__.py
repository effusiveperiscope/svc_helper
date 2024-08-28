from svc_helper.svc.rvc.configs.config import Config
from svc_helper.svc.rvc.modules.vc.modules import VC
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def find_adjacent_index(model_path):
    if not os.path.exists(model_path):
        return None
    parent = Path(model_path).parent
    index_search = glob.glob(parent / '*.index')
    return index_search[0]

class RVCModel:
    expected_sample_rate = 16000
    def __init__(self):
        lib_dir = os.path.dirname(os.path.abspath(__file__))
        config = Config(lib_dir)
        vc = VC(config)
        self.vc = vc
        self.index_path = ''

    def load_model(self, model_path, index_path=None):
        t = self.vc.get_vc(model_path)
        if index_path is None:
            self.index_path = find_adjacent_index(model_path)
        else:
            self.index_path = index_path

    def output_sample_rate(self):
        return self.vc.tgt_sr

    """
    extra hooks is a dict containing optional hooks:
        'feature_transform' is a function accepting the features tensor
        allowing you to transform features inplace
        'feature_override' accepts the padded audio input in RVC to
        output replacement features

    All other settings are as used in RVC """
    def infer_file(self,
        input_path,
        transpose=0,
        f0_file=None,
        f0_method='rmvpe',
        index_rate=0.0,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1,
        protect=0.33,
        extra_hooks={},
        target_pitch=None):
        status, (tgt_sr, wav_opt) = self.vc.vc_single(
            sid=0,
            input_audio_path=input_path,
            f0_up_key=transpose,
            f0_file=f0_file,
            f0_method=f0_method,
            file_index=self.index_path,
            file_index2=self.index_path,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            extra_hooks=extra_hooks,
            target_pitch=target_pitch
        )
        return wav_opt

    """
    Version of infer that works with audio data in memory
    """
    def infer_audio(self,
        input_audio,
        transpose=0,
        f0_file=None,
        f0_method='rmvpe',
        index_rate=0.0,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1,
        protect=0.33,
        extra_hooks={},
        target_pitch=None):
        status, (tgt_sr, wav_opt) = self.vc.vc_audio(
            sid=0,
            input_audio_array=input_audio,
            f0_up_key=transpose,
            f0_file=f0_file,
            f0_method=f0_method,
            file_index=self.index_path,
            file_index2=self.index_path,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            extra_hooks=extra_hooks,
            target_pitch=target_pitch
        )
        return wav_opt

    def check_initialized(self):
        return self.vc.pipeline is not None and self.vc.net_g is not None

    """
    Perform RVC f0 transformations on an existing pitch curve
    """
    def f0_transform(self, f0, f0_up_key=0):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 *= pow(2, f0_up_key / 12)

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    """
    Raw RVC inference without windowing, protection, index ratio
    """
    def raw_infer(self,
        feats : np.ndarray,
        pitch : np.ndarray,
        pitchf : np.ndarray,
        audio : np.ndarray = None, sid=0):
        pipeline = self.vc.pipeline
        net_g = self.vc.net_g
        hasp = pitch is not None and pitchf is not None

        assert self.check_initialized(), 'Model not loaded'

        pitch = torch.from_numpy(pitch).to(
            pipeline.device).unsqueeze(0).long()
        pitchf = torch.from_numpy(pitchf).to(
            pipeline.device).unsqueeze(0).float()
        if type(feats) == np.ndarray:
            feats = torch.from_numpy(feats).to(pipeline.device)
        else:
            feats = feats.to(pipeline.device)

        # (Why does the network take lerped feature inputs?)
        feats = F.interpolate(
            feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if audio is not None:
            p_len = audio.shape[0] // pipeline.window
            p_len = min(p_len, feats.shape[1])
        else:
            p_len = feats.shape[1]
        p_len = torch.tensor([p_len], device=pipeline.device).long()
        if hasp:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

        sid = torch.tensor(sid, device=pipeline.device).unsqueeze(0).long()

        if pipeline.is_half:
            pitchf = pitchf.half()
            feats = feats.half()

        arg = (feats, p_len, pitch, pitchf, sid) if hasp else (
            feats, p_len, sid)
        with torch.no_grad():
            audio_result = (
                net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_result