from svc_helper.svc.rvc.configs.config import Config
from svc_helper.svc.rvc.modules.vc.modules import VC
import os
import glob
from pathlib import Path

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