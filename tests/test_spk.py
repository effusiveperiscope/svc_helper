from svc_helper.speaker.models import SVC5SpeakerEncoder

def test_speaker_encoder():
    svc5_spk_encoder = SVC5SpeakerEncoder(
        device='cuda')
    feat = svc5_spk_encoder.extract_feature('tests/ood5_male.wav')
    print(feat.shape)

test_speaker_encoder()