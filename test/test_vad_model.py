from funasr import AutoModel
from pathlib import Path

MODEL_DIR = Path("D:\Project\WiseHome\model")
VAD_MODEL = "fsmn-vad"
ASR_MODEL = "SenseVoiceSmall"   # 改成流式模型
VAD_MODEL_PATH = MODEL_DIR / VAD_MODEL
ASR_MODEL_PATH = MODEL_DIR / ASR_MODEL

asr_model = AutoModel(
    model = ASR_MODEL_PATH,
    vad_model=VAD_MODEL_PATH,
    disable_update=True
)

vad_model = AutoModel(
    model = VAD_MODEL_PATH,
    disable_update=True
)


res = asr_model.generate(
    input="output_custom_voice.wav",
    cache={},
    vad_model=vad_model,
    is_final=False,
    disable_pbar=True,
)

print(res)