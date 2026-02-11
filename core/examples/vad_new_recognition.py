import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from funasr import AutoModel
from pathlib import Path

# 模型缓存目录；如果能够获取到缓存则使用缓存目录中的模型，否则从互联网下载模型
MODEL_CACHE_DIR = "model"

VAD_MODEL = "SenseVoiceSmall"
ASR_MODEL = "fsmn-vad"

VAD_DIR = Path(MODEL_CACHE_DIR) / VAD_MODEL
ASR_DIR = Path(MODEL_CACHE_DIR) / ASR_MODEL

SAMPLE_RATE = 16000
CHUNK_SIZE = 9600
CHUNK_SIZE_MS = CHUNK_SIZE * 1000 / SAMPLE_RATE

CHANNELS = 1


class VoiceRecognizer:
    def __init__(self):
        # 创建 VAD 模型实例
        self.vad_model = AutoModel(model=VAD_DIR, device="cuda:0", disable_pbar=True)
        # 创建 ASR 模型实例
        self.asr_model = AutoModel(model=ASR_DIR, device="cuda:0", disable_pbar=True)

        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.chunk_size_ms = CHUNK_SIZE_MS

        self.cache_vad = {}
        self.cache_asr = {}

        self.last_vad_beg = -1
        self.last_vad_end = -1

        self.offset = 0
        self.segment_start_time = None

        self.audio_buffer = np.array([], dtype=np.float32)

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调"""
        if status:
            print(f"音频状态出错: {status}")
        
        self.audio_buffer = np.append(self.audio_buffer, indata[:, 0])

    def _recognize(self, audio: np.ndarray) -> str:
        """通过 ASR 模型识别语音"""
        try:
            result = self.asr_model.generate(
                input=audio,
                cache=self.cache_asr,
                language="zh",
                batch_size=1
            )
            
            if result and len(result) > 0 and 'text' in result[0]:
                return result[0]['text'].strip()
            
            return ""
            
        except Exception as e:
            print(f"ASR 模型识别错误: {e}")
            return ""

    def _get_segment(self, res):
        if len(res[0]["value"]):
            vad_segments = res[0]["value"]
            for segment in vad_segments:
                # speech begin
                if segment[0] > -1:
                    self.last_vad_beg = segment[0]
                    if self.segment_start_time is None:
                        print("[segment] 语音开始")

                # speech end
                if segment[1] > -1:
                    self.last_vad_end = segment[1]

                if self.last_vad_beg > -1 and self.last_vad_end > -1:
                    self.last_vad_beg -= self.offset
                    self.last_vad_end -= self.offset
                    self.offset += self.last_vad_end

                    beg = int(self.last_vad_beg * self.sample_rate / 1000)
                    end = int(self.last_vad_end * self.sample_rate / 1000)

                    return beg, end

    def get_voice_input(self):
        """获取语音输入"""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    if len(audio_buffer) < self.chunk_size:
                        time.sleep(0.01)
                        continue

                    chunk = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size:]
                    audio_vad = np.append(audio_vad, chunk)

                    beg, end = self._get_segment(self.vad_model.generate(
                        input=chunk,
                        cache=self.cache_vad,
                        is_final=False,
                        chunk_size=self.chunk_size_ms
                    ))

                    result = self._recognize(audio_vad[beg:end])

            except Exception as e:
                print(f"发生错误: {e}")

# 全局实例
_recognizer = None

def get_recognizer() -> VoiceRecognizer:
    """获取处理器单例"""
    global _recognizer
    if _recognizer is None:
        _recognizer = VoiceRecognizer()
    return _recognizer

async def get_voice_input() -> str:
    recognizer = get_recognizer()
    return await recognizer.get_voice_input()

async def main():
    while True:
        text = await get_voice_input()
        print(f"识别结果: {text}")

if __name__ == "__main__":
    asyncio.run(main())