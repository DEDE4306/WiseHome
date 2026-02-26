import asyncio
import logging
import re
import time

import numpy as np
import sounddevice as sd
from funasr import AutoModel

from core.tts import tts_speech
from core.logger import logger

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
            keywords="你好小爱",
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
        self.last_speech_time = 0

        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_vad = np.array([], dtype=np.float32)

        self.activated = False  # 是否进入唤醒状态


    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.error("发生错误：", status)
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

        # res 的输出类似于:
        # [{'key': 'output_custom_voice', 'text': 'detected 你好小爱 0.30402980562956583', 'text2': 'detected 你好小爱 0.46939074955284443'}]
        # [{'key': 'output_custom_voice', 'text': 'rejected', 'text2': 'rejected'}]

        if not res:
            return False

        print("kws 输出：", res)

        text = res[0].get("text", "")
        text2 = res[0].get("text2", "")

        self.cache_kws = {}

        if text.startswith("detected") or text2.startswith("detected"):
            print("text 内容：", text)
            print("text2 内容", text2)
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
            logger.info("等待唤醒词...")
            while True:
                if len(self.audio_buffer) < self.chunk_size:
                    time.sleep(0.01)
                    continue

                chunk = self.audio_buffer[:self.chunk_size]
                self.audio_buffer = self.audio_buffer[self.chunk_size:]

                self.audio_vad = np.append(self.audio_vad, chunk)

                # vad 检测
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

                        print(f"vad 时间：{segment[0]}--{segment[1]}")

                        # 检测到语音开始
                        if segment[0] > -1:
                            # 如果之前有未处理的语音段，先重置
                            if self.last_vad_beg > -1 and self.last_vad_end > -1:
                                print("警告：检测到新语音开始，但前一段未处理完成，重置状态")
                            self.last_vad_beg = segment[0]
                            if self.segment_start_time is None:
                                self.segment_start_time = time.time()

                        # 检测到语音结束
                        if segment[1] > -1:
                            self.last_vad_end = segment[1]

                        # 当语音开始和结束都检测到时，处理该段
                        if self.last_vad_beg > -1 and self.last_vad_end > -1:
                            print("处理前：")
                            print("offset: ", self.offset)
                            print("last_vad_beg: ", self.last_vad_beg)
                            print("last_vad_end: ", self.last_vad_end)

                            # 计算相对时间
                            beg_relative = self.last_vad_beg - self.offset
                            end_relative = self.last_vad_end - self.offset
                            
                            # 更新 offset
                            self.offset += end_relative

                            print("处理后：")
                            print("offset: ", self.offset)
                            print("beg_relative: ", beg_relative)
                            print("end_relative: ", end_relative)

                            beg = int(beg_relative * self.sample_rate / 1000)
                            end = int(end_relative * self.sample_rate / 1000)

                            print(f"[vad] 语音段长度: {end - beg} samples")

                            # 检查索引是否有效
                            if beg < 0 or end > len(self.audio_vad):
                                print(f"警告：音频段索引越界 [{beg}, {end}]，音频长度: {len(self.audio_vad)}")
                                # 重置状态
                                self.last_vad_beg = self.last_vad_end = -1
                                continue

                            # 语音唤醒
                            speech_segment = self.audio_vad[beg:end]

                            if not self.activated:
                                kws_hit = self._check_kws(speech_segment)

                                if kws_hit:
                                    # 未命中唤醒词，丢弃该段
                                    print("关键词命中")
                                    print("在的，主人")
                                    # tts_speech("在的，主人")
                                    self.activated = True
                                    print("语音已激活")
                                    self.last_speech_time = time.time()
                                    continue
                            else:
                                if len(speech_segment) > 0:
                                    result = self.asr_model.generate(
                                        speech_segment,
                                        lang="auto",
                                        cache=self.cache_asr,
                                        disable_pbar=True,
                                        is_final=True
                                    )

                                    if result:
                                        text = self._clean_funasr_output(result)
                                        print(f"[ASR] {text}")

                                        self.audio_vad = self.audio_vad[end:]
                                        self.last_vad_beg = self.last_vad_end = -1
                                        self.segment_start_time = None
                                        self.activated = False  # 处理完毕，重新进入等待唤醒
                                        print("等待唤醒词...")
                                        return text

                            # 处理完该段后，只保留未处理的音频
                            self.audio_vad = self.audio_vad[end:]
                            # 重置状态
                            self.last_vad_beg = self.last_vad_end = -1
                            self.segment_start_time = None
                            
                            self.last_speech_time = time.time()

                if self.activated and (time.time() - self.last_speech_time > 10):
                    print("AI: 没有听到主人说话。如果需要，请随时叫我。")
                    # tts_speech("没有听到主人说话。如果需要，请随时叫我。")
                    self.activated = False
                    self.audio_vad = np.array([], dtype=np.float32)

                time.sleep(0.001)

# 全局语音处理器
_recognizer = None

def get_recognizer() -> VoiceRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = VoiceRecognizer()
    return _recognizer

async def main():
    while True:
        recognizer = get_recognizer()
        text = await recognizer.get_voice_input()
        print(f"识别结果: {text}")

if __name__ == "__main__":
    asyncio.run(main())
