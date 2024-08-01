from svc_helper.svc.rvc import RVCModel
from svc_helper.sfeatures.models import RVCHubertModel
from huggingface_hub import hf_hub_download
import librosa
import soundfile as sf
import torch

OUTPUT_FILES = True

def test_rvc():
    rvc_model = RVCModel()
    input_path = 'tests/test_speech.wav'

    test_model_path = hf_hub_download(repo_id='therealvul/RVCv2', 
        filename='RainbowDashS1/RainbowDashS1.pth')
    test_index_path = hf_hub_download(repo_id='therealvul/RVCv2', 
        filename='RainbowDashS1/added_IVF1357_Flat_nprobe_1_RainbowDashS1_v2.index')

    rvc_model.load_model(model_path = test_model_path,
        index_path = test_index_path)

    wav_opt = rvc_model.infer_file(input_path, transpose=12)
    if OUTPUT_FILES:
        sf.write('tests/test_rvc_output_1.wav', wav_opt,
            rvc_model.output_sample_rate())

    # Try feature replacement
    # data, rate = librosa.load('tests/test_speech.wav',
    #     sr=RVCHubertModel.expected_sample_rate)
    # rvc_hubert = RVCHubertModel()
    # feat = rvc_hubert.extract_features(torch.from_numpy(data))

    # Add random noise
    wav_opt = rvc_model.infer_file(input_path, transpose=12,
        extra_hooks={
        'feature_transform': lambda t: t + torch.randn_like(t)*0.5})
    if OUTPUT_FILES:
        sf.write('tests/test_rvc_output_2.wav', wav_opt,
            rvc_model.output_sample_rate())

    # Test pitch targeting
    wav_opt = rvc_model.infer_file(input_path, transpose=0,
        target_pitch=400)
    if OUTPUT_FILES:
        sf.write('tests/test_rvc_output_3.wav', wav_opt,
            rvc_model.output_sample_rate())

    # Test from audio
    audio, _ = librosa.load(input_path, sr=rvc_model.expected_sample_rate)
    wav_opt = rvc_model.infer_file(input_path, transpose=12)
    if OUTPUT_FILES:
        sf.write('tests/test_rvc_output_4.wav', wav_opt,
            rvc_model.output_sample_rate())