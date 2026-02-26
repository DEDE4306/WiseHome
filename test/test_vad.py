
from funasr import AutoModel
from pathlib import Path
from config.constants import VAD_MODEL_PATH



vad_model = AutoModel(
    model = VAD_MODEL_PATH,
    disable_update=True
)

res = vad_model.generate(
    input="silence.wav",
    cache={},
    vad_model=vad_model,
    disable_pbar=True,
)

print(res)