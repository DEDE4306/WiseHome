import time
import json
import numpy as np
import sounddevice as sd
import logging as logger
from funasr import AutoModel

# 假定以下对象与你原工程中一致
# model_vad
# asr
# format_str_v3
# config
# logger

model_vad = AutoModel(model="fsmn-vad")

model_asr = AutoModel(model="iic/SenseVoiceSmall")

sample_rate = 16000
chunk_size_ms = 300
chunk_size = int(chunk_size_ms * sample_rate / 1000)    # 分片大小 300ms

audio_buffer = np.array([], dtype=np.float32)
audio_vad = np.array([], dtype=np.float32)

cache_vad = {}
cache_asr = {}

last_vad_beg = last_vad_end = -1
offset = 0
segment_start_time = None


def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        logger.warning(status)
    audio_buffer = np.append(audio_buffer, indata[:, 0])


logger.info("开始麦克风实时识别，Ctrl+C 退出")

with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    blocksize=chunk_size,
    callback=audio_callback,
):
    try:
        while True:
            if len(audio_buffer) < chunk_size:
                time.sleep(0.01)
                continue

            chunk = audio_buffer[:chunk_size]
            audio_buffer = audio_buffer[chunk_size:]
            audio_vad = np.append(audio_vad, chunk)

            # ---------- VAD ----------
            res = model_vad.generate(
                input=chunk,
                cache=cache_vad,
                is_final=False,
                chunk_size=chunk_size_ms
            )

            if len(res[0]["value"]):
                vad_segments = res[0]["value"]
                for segment in vad_segments:
                    # speech begin
                    if segment[0] > -1:
                        last_vad_beg = segment[0]
                        if segment_start_time is None:
                            segment_start_time = time.time()
                            logger.info("[segment] 语音开始")

                    # speech end
                    if segment[1] > -1:
                        last_vad_end = segment[1]

                    if last_vad_beg > -1 and last_vad_end > -1:
                        last_vad_beg -= offset
                        last_vad_end -= offset
                        offset += last_vad_end

                        beg = int(last_vad_beg * sample_rate / 1000)
                        end = int(last_vad_end * sample_rate / 1000)

                        logger.info(f"[vad] 语音段长度: {end - beg} samples")

                        # ---------- ASR ----------
                        result = model_asr.generate(
                            audio_vad[beg:end],
                            lang="auto",
                            cache=cache_asr,
                            is_final=True
                        )

                        if segment_start_time is not None:
                            elapsed = (time.time() - segment_start_time) * 1000
                            logger.info(f"[segment] 识别耗时: {elapsed:.2f} ms")

                        if result is not None:
                            text = result[0]["text"]
                            logger.info(f"[ASR] {text}")
                            print(text)

                        # ---------- reset ----------
                        audio_vad = audio_vad[end:]
                        last_vad_beg = last_vad_end = -1
                        segment_start_time = None

            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("录音结束，退出")
