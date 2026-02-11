import sounddevice as sd
import numpy as np
from funasr import AutoModel
import os
import soundfile as sf

sr = 16000
channel = 1

chunk_size = [0, 20, 5]
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

all_audio = []
chunk_count = 0


model = AutoModel(
    model="fsmn-vad",
    disable_pbar=True
)
cache = {}


def callback(indata, frames, time, status):
    global all_audio, chunk_count
    if status:
        print("Status:", status)

    audio = indata[:, 0]
    audio = np.array(audio)
    
    all_audio.append(audio)

    volume = np.abs(audio).mean()
    volume_db = 20 * np.log10(volume + 1e-10)

    chunk_count += 1

    try:
        res = model.generate(
            input=audio,
            cache=cache,
            is_final=False,
            chunk_size=9600,
        )
    except Exception as e:
        print(f"识别错误: {e}")


print("开始录音，按 Ctrl+C 停止")

try:
    with sd.InputStream(
            samplerate=sr,
            channels=channel,
            dtype="float32",
            callback=callback,
            blocksize=19200
    ):
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\n录音结束")
