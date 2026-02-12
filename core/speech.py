import asyncio
import re
import time
import logging as logger

import numpy as np
import sounddevice as sd
from funasr import AutoModel

SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 300
CHUNK_SIZE = int(CHUNK_SIZE_MS * SAMPLE_RATE / 1000)

CHANNELS = 1


class VoiceRecognizer:
    def __init__(self):
        self.vad_model = AutoModel(model="fsmn-vad")
        self.asr_model = AutoModel(model="iic/SenseVoiceSmall")

        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.chunk_size_ms = CHUNK_SIZE_MS
        self.channels = CHANNELS

        self.cache_vad = {}
        self.cache_asr = {}

        self.last_vad_beg = -1
        self.last_vad_end = -1

        self.offset = 0
        self.segment_start_time = None

        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_vad = np.array([], dtype=np.float32)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(status)
        self.audio_buffer = np.append(self.audio_buffer, indata[:, 0])

    def _clean_funasr_output(self, funasr_result_list):
        """
        funasr_result_list: 模型返回的结果列表，每个元素是 dict，例如：
            {"text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>好，你最近一次量体重是什么时间？"}
        返回拼接后的干净文本
        """
        texts = []
        for item in funasr_result_list:
            text = item["text"]
            # 去掉所有尖括号标记
            text = re.sub(r"<\|.*?\|>", "", text)
            text = text.strip()
            if text:
                texts.append(text)
        # 用空格或者换行拼接
        return " ".join(texts)

    async def get_voice_input(self) -> str:
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            while True:
                if len(self.audio_buffer) < self.chunk_size:
                    time.sleep(0.01)
                    continue

                chunk = self.audio_buffer[:self.chunk_size]
                self.audio_buffer = self.audio_buffer[self.chunk_size:]
                self.audio_vad = np.append(self.audio_vad, chunk)

                res = self.vad_model.generate(
                    input=chunk,
                    cache=self.cache_vad,
                    is_final=False,
                    disable_pbar=True,
                    chunk_size=self.chunk_size_ms

                )

                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:
                            self.last_vad_beg = segment[0]
                            if self.segment_start_time is None:
                                self.segment_start_time = time.time()

                        if segment[1] > -1:
                            self.last_vad_end = segment[1]

                        if self.last_vad_beg > -1 and self.last_vad_end > -1:
                            self.last_vad_beg -= self.offset
                            self.last_vad_end -= self.offset
                            self.offset += self.last_vad_end

                            beg = int(self.last_vad_beg * self.sample_rate / 1000)
                            end = int(self.last_vad_end * self.sample_rate / 1000)

                            # print(f"[vad] 语音段长度: {end - beg} samples")

                            result = self.asr_model.generate(
                                self.audio_vad[beg:end],
                                lang="auto",
                                cache=self.cache_asr,
                                disable_pbar=True,
                                is_final=True
                            )

                            if self.segment_start_time is not None:
                                elapsed = (time.time() - self.segment_start_time) * 1000
                                # print(f"[segment] 识别耗时: {elapsed:.2f} ms")

                            if result is not None:
                                text = self._clean_funasr_output(result)
                                # print(f"[ASR] {text}")

                                self.audio_vad = self.audio_vad[end:]
                                self.last_vad_beg = self.last_vad_end = -1
                                self.segment_start_time = None
                                return text

                time.sleep(0.001)


_recognizer = None

def get_recognizer() -> VoiceRecognizer:
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
