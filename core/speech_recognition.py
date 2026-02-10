import asyncio
import sounddevice as sd
import numpy as np
from collections import deque
from funasr import AutoModel


sr = 16000
channel = 1
block_size = 9600

VAD_THRESHOLD_START = -40
VAD_THRESHOLD_END = -50
VAD_WINDOW_SIZE = 15
VAD_START_RATIO = 0.7
VAD_END_RATIO = 0.9


class SimpleVoiceRecognizer:
    def __init__(self):
        self.model = AutoModel(model="D:\Project\WiseHome\model\paraformer-zh-streaming")
        self.cache = {}
        self.audio_buffer = []
        self.volume_window = deque(maxlen=VAD_WINDOW_SIZE)
        self.is_recording = False
        self.recognition_event = asyncio.Event()
        self.recognition_result = ""
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("音频状态异常:", status)

        audio = indata[:, 0]
        volume = np.abs(audio).mean()
        volume_db = 20 * np.log10(volume + 1e-10)
        self.volume_window.append(volume_db)

        if self.is_recording:
            self.audio_buffer.append(audio)
            if self._check_voice_end():
                self._stop_recording()
        else:
            if self._check_voice_start():
                self._start_recording(audio)

    def _check_voice_start(self):
        if len(self.volume_window) < VAD_WINDOW_SIZE:
            return False
        count = sum(1 for v in self.volume_window if v > VAD_THRESHOLD_START)
        return count / VAD_WINDOW_SIZE >= VAD_START_RATIO

    def _check_voice_end(self):
        if len(self.volume_window) < VAD_WINDOW_SIZE:
            return False
        count = sum(1 for v in self.volume_window if v < VAD_THRESHOLD_END)
        return count / VAD_WINDOW_SIZE >= VAD_END_RATIO

    def _start_recording(self, audio):
        self.is_recording = True
        self.audio_buffer = [audio]
        print("检测到语音开始，开始录音...")

    def _stop_recording(self):
        self.is_recording = False
        print("检测到语音结束，开始识别...")
        asyncio.create_task(self._recognize())

    async def _recognize(self):
        if not self.audio_buffer:
            self.recognition_result = ""
            self.recognition_event.set()
            return

        full_audio = np.concatenate(self.audio_buffer)
        try:
            res = self.model.generate(
                input=full_audio,
                cache=self.cache,
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            self.recognition_result = res[0]['text']
            print(f"识别结果: {self.recognition_result}")
        except Exception as e:
            print(f"识别错误: {e}")
            self.recognition_result = ""
        finally:
            self.recognition_event.set()

    async def get_voice_input(self) -> str:
        self.recognition_event.clear()
        self.recognition_result = ""
        
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=sr,
                channels=channel,
                dtype="float32",
                callback=self.audio_callback,
                blocksize=block_size,
            )
            self.stream.start()
        
        await self.recognition_event.wait()
        return self.recognition_result

    async def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


_recognizer = None


async def get_voice_input() -> str:
    global _recognizer
    if _recognizer is None:
        _recognizer = SimpleVoiceRecognizer()
    return await _recognizer.get_voice_input()


async def close_voice_recognizer():
    global _recognizer
    if _recognizer is not None:
        await _recognizer.close()
        _recognizer = None

async def main():
    while True:
        text = await get_voice_input()
        print(text)

if __name__ == "__main__":
    asyncio.run(main())