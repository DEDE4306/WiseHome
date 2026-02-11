# voice_input.py
import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from funasr import AutoModel

# 全局配置
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 9600  # 约 0.6 秒 (9600 / 16000)

# 全局模型（懒加载）
_recognizer = None


class SimpleVoiceRecognizer:
    def __init__(self):
        print("正在加载语音识别模型...")
        self.model = AutoModel(
            model="D:/Project/WiseHome/model/paraformer-zh-streaming",
            # punc_model="ct-punc",    # 可选：加标点（需下载）
            disable_pbar=True          # 关闭进度条
        )
        self.cache = {}
        self.audio_queue = queue.Queue()
        self.result_text = ""
        self.final_result = ""
        self.is_listening = False
        self.stream = None

    async def get_voice_input(self) -> str:
        """启动一次语音输入，返回识别出的完整句子"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_and_recognize)

    def _record_and_recognize(self) -> str:
        self.final_result = ""
        self.is_listening = True

        # 启动录音流
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=BLOCK_SIZE
        ):
            print("🎤 请说话（说完后停顿1秒）...")
            # 等待语音结束（由 VAD 决定何时 is_final=True）
            while self.is_listening:
                sd.sleep(100)  # 每 100ms 检查一次

        print(f"✅ 识别完成: '{self.final_result}'")
        return self.final_result.strip()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status)

        audio_chunk = indata[:, 0]

        try:
            # 调用 FunASR 流式识别（自动用 VAD 判断是否结束）
            res = self.model.generate(
                input=audio_chunk,
                cache=self.cache,
                is_final=False,  # FunASR 内部会根据 VAD 决定是否输出 final
                chunk_size=[0, 10, 5],  # 可微调，[0, 10, 5] 更灵敏
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )

            # FunASR 的 res 是 list of dict
            if res and len(res) > 0:
                text = res[0].get("text", "").strip()
                if text:
                    self.result_text = text
                    # 检查是否为最终结果（FunASR 在 VAD 结束后会清空 cache 或标记 final）
                    # 实际上，当 VAD 检测到静音，下一次 generate 会返回完整句子并重置状态
                    # 我们用一个 trick：如果连续几块无新内容，就认为结束了
                    # 但更简单的方式：等 cache 被清空（或看是否有 is_final 标志）

                # 【关键】判断是否为最终输出：当 cache 中的 key 被重置 或 文本长时间不变
                # FunASR 官方 demo 中通常靠外部 VAD 控制，但我们这里依赖其内部 VAD 行为
                # 实测：当语音结束后，再送一块静音，会触发 is_final 效果

                # 简单策略：如果当前文本非空，且下一块是静音（音量低），则结束
                volume = np.abs(audio_chunk).mean()
                if volume < 0.01 and self.result_text:
                    # 再送一块静音触发 final
                    silent_chunk = np.zeros_like(audio_chunk)
                    final_res = self.model.generate(
                        input=silent_chunk,
                        cache=self.cache,
                        is_final=True,
                        chunk_size=[0, 10, 5],
                        encoder_chunk_look_back=4,
                        decoder_chunk_look_back=1
                    )
                    if final_res and final_res[0].get("text", "").strip():
                        self.final_result = final_res[0]["text"].strip()
                    else:
                        self.final_result = self.result_text
                    self.is_listening = False

        except Exception as e:
            print(f"识别错误: {e}")
            self.is_listening = False


# 全局实例
_recognizer = None

async def get_voice_input() -> str:
    global _recognizer
    if _recognizer is None:
        _recognizer = SimpleVoiceRecognizer()
    return await _recognizer.get_voice_input()

async def main():
    while True:
        text = await get_voice_input()
        print(text)

if __name__ == "__main__":
    asyncio.run(main())