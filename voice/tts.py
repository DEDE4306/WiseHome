import torch
import sounddevice as sd
import numpy as np
from qwen_tts import Qwen3TTSModel

from config.constants import TTS_MODEL_PATH

model = Qwen3TTSModel.from_pretrained(
    TTS_MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)


def tts_speech(speech: str):
    wav, sr = model.generate_custom_voice(
        text=speech,
        language="Chinese",
        speaker="Vivian",
        instruct="你是一个智能语音助手，请使用礼貌而温柔的语气",
    )

    # 确保 wav 是一维数组（单声道）
    if isinstance(wav, list):
        wav = np.array(wav)
    if wav.ndim == 2:
        wav = wav.squeeze()  # 如果是 (1, N) → (N,)

    # 播放音频
    sd.play(wav, samplerate=sr)
    sd.wait()  # 等待播放完成