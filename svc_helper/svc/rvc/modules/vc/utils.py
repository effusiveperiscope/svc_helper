import os

from fairseq import checkpoint_utils
from huggingface_hub import hf_hub_download


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    rvc_hubert_path = hf_hub_download(
        repo_id='therealvul/svc_helper', filename='rvc_hubert.pt')
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [rvc_hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
