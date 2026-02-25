import asyncio
import re
import time

import numpy as np
import sounddevice as sd
from funasr import AutoModel

from config.constants import *


class VoiceRecognizer:
    def __init__(self):
        # vad 检测模型
        self.vad_model = AutoModel(
            model=VAD_MODEL_PATH,
            disable_update=True
        )
        # 语音识别 asr 模型
        self.asr_model = AutoModel(
            model=ASR_MODEL_PATH,
            disable_update=True
        )
        # 关键词检测模型
        self.kws_model = AutoModel(
            model=KWS_MODEL_PATH,
            keywords=KEYWORD,
            disable_update=True,
            output_dir="./outputs"
        )

        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.chunk_size_ms = CHUNK_SIZE_MS
        self.channels = CHANNELS

        self.cache_vad = {}
        self.cache_asr = {}
        self.cache_kws = {}

        self.last_vad_beg = -1
        self.last_vad_end = -1
        self.offset = 0
        self.segment_start_time = None

        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_vad = np.array([], dtype=np.float32)

        self.activated = False  # 唤醒状态标志

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("发生错误：", status)
        self.audio_buffer = np.append(self.audio_buffer, indata[:, 0])

    def _clean_funasr_output(self, funasr_result_list):
        texts = []
        # funasr_result_list: 模型返回的结果列表，每个元素是 dict
        # {"text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好"}
        for item in funasr_result_list:
            text = item.get("text")
            # 去掉所有尖括号标记
            text = re.sub(r"<\|.*?\|>", "", text)
            # 去除空格
            text = text.strip()
            if text:
                texts.append(text)
        # 用空格或者换行拼接
        return " ".join(texts)

    def _check_kws(self, chunk) -> bool:
        """对当前 chunk 做关键词检测，返回是否命中"""
        res = self.kws_model.generate(
            input=chunk,
            cache=self.cache_kws,
            disable_pbar=True,
        )
        # res[0]["value"] 为检测到的关键词列表，非空则命中
        if res and res[0].get("text2", "") != "rejected" and res[0].get("text2", "") != "":
            text2 = res[0].get("text2", "")
            print("text2 内容：",text2)
            # text2 格式: "detected <关键词> <置信度>"
            parts = text2.split()
            keyword = parts[1] if len(parts) >= 2 else "unknown"
            score = float(parts[2]) if len(parts) >= 3 else 0.0
            print(f"[KWS] 检测到唤醒词：{keyword}，置信度：{score:.4f}")
            self.cache_kws = {}
            return True

        return False

    async def get_voice_input(self) -> str:
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            print("等待唤醒词...")
            while True:
                if len(self.audio_buffer) < self.chunk_size:
                    time.sleep(0.01)
                    continue

                chunk = self.audio_buffer[:self.chunk_size]
                self.audio_buffer = self.audio_buffer[self.chunk_size:]

                if not self.activated:
                    if self._check_kws(chunk):
                        self.activated = True
                        # 重置 VAD/ASR 状态，准备接收后续指令
                        self.cache_vad = {}
                        self.cache_asr = {}
                        self.audio_vad = np.array([], dtype=np.float32)
                        self.last_vad_beg = self.last_vad_end = -1
                        self.offset = 0
                        self.segment_start_time = None
                        print("已唤醒，请说话...")
                    await asyncio.sleep(0.001)
                    continue

                self.audio_vad = np.append(self.audio_vad, chunk)

                res = self.vad_model.generate(
                    input=chunk,
                    cache=self.cache_vad,
                    is_final=False,
                    disable_pbar=True,
                    chunk_size=self.chunk_size_ms
                )

                print("vad 原始输出：", res)

                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]

                    print("vad 段：", vad_segments)
                    for segment in vad_segments:

                        print("vad 时间：" + segment[0] + "--" + segment[1])

                        if segment[0] > -1:
                            self.last_vad_beg = segment[0]
                            if self.segment_start_time is None:
                                self.segment_start_time = time.time()

                        if segment[1] > -1:
                            self.last_vad_end = segment[1]

                        if self.last_vad_beg > -1 and self.last_vad_end > -1:
                            print("处理前：")
                            print("offset: ", self.offset)
                            print("last_vad_beg: ", self.last_vad_beg)
                            print("last_vad_end: ", self.last_vad_end)

                            self.last_vad_beg -= self.offset
                            self.last_vad_end -= self.offset
                            self.offset += self.last_vad_end

                            print("处理后：")
                            print("offset: ", self.offset)
                            print("last_vad_beg: ", self.last_vad_beg)
                            print("last_vad_end: ", self.last_vad_end)

                            beg = int(self.last_vad_beg * self.sample_rate / 1000)
                            end = int(self.last_vad_end * self.sample_rate / 1000)

                            # print(f"[vad] 语音段长度: {end - beg} samples")

                            # 语音唤醒

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

                            if result:
                                text = self._clean_funasr_output(result)
                                # print(f"[ASR] {text}")

                                self.audio_vad = self.audio_vad[end:]
                                self.last_vad_beg = self.last_vad_end = -1
                                self.segment_start_time = None
                                self.activated = False  # 处理完毕，重新进入等待唤醒
                                print("等待唤醒词...")
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
