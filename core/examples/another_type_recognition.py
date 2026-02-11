import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from funasr import AutoModel
from pathlib import Path
from typing import List, Tuple, Optional

# 模型缓存目录；如果能够获取到缓存则使用缓存目录中的模型，否则从互联网下载模型
MODEL_CACHE_DIR = "model"

VAD_MODEL = "fsmn-vad"
ASR_MODEL = "SenseVoiceSmall"

VAD_DIR = Path(MODEL_CACHE_DIR) / VAD_MODEL
ASR_DIR = Path(MODEL_CACHE_DIR) / ASR_MODEL

# 音频配置
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 9600  # 约 0.6 秒 (9600 / 16000)

# VAD 配置
VAD_CHUNK_SIZE = 9600  # VAD 模型块大小
MAX_RECORDING_DURATION = 30.0  # 最大录音时长（秒）
SILENCE_AFTER_SPEECH = 1.0  # 语音结束后的静音等待时间（秒）


class VoiceRecognizer:
    def __init__(self):
        print("正在加载 VAD 模型...")
        self.vad_model = AutoModel(model=VAD_DIR, disable_pbar=True)
        print("正在加载 ASR 模型...")
        self.asr_model = AutoModel(model=ASR_DIR, disable_pbar=True)
        print("模型加载完成！")
        
        self.cache = {}
        self.audio_buffer = []
        self.vad_segments = []
        self.is_listening = False
        self.recording_start_time = None
        
        # 异步处理队列
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.processing_stop_event = threading.Event()

    async def get_voice_input(self) -> str:
        """启动一次语音输入，返回识别出的完整句子"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_and_recognize)

    def _record_and_recognize(self) -> str:
        """录音并识别语音"""
        self.is_listening = True
        self.audio_buffer = []
        self.vad_segments = []
        self.recording_start_time = None
        self.cache = {}
        
        # 启动处理线程
        self.processing_stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()

        # 启动录音流
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=BLOCK_SIZE
        ):
            print("🎤 请说话...")
            # 等待语音结束
            while self.is_listening:
                sd.sleep(100)

        # 停止处理线程
        self.processing_stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2)

        # 提取语音片段并识别
        return self._extract_and_recognize()

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调"""
        if status:
            if status.input_overflow:
                if not hasattr(self, '_overflow_logged'):
                    print("警告: 音频缓冲区溢出，考虑增大 blocksize 或减少处理量")
                    self._overflow_logged = True
            else:
                print(f"Audio status: {status}")

        audio_chunk = indata[:, 0]
        
        # 将音频块放入队列，异步处理
        self.audio_queue.put({
            'audio_chunk': audio_chunk,
            'timestamp': time.currentTime
        })

    def _processing_worker(self):
        """后台处理线程 - 处理 VAD 检测逻辑"""
        while not self.processing_stop_event.is_set():
            try:
                # 从队列获取数据，超时 0.1 秒
                data = self.audio_queue.get(timeout=0.1)
                
                audio_chunk = data['audio_chunk']
                current_time = data['timestamp']
                
                # 保存音频到缓冲区
                self.audio_buffer.append(audio_chunk)
                
                # 记录录音开始时间
                if self.recording_start_time is None:
                    self.recording_start_time = current_time
                
                # 使用 VAD 模型检测语音端点
                self._detect_vad_segments(audio_chunk, current_time)
                
                # 检查是否应该结束录音
                self._check_recording_end(current_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程错误: {e}")

    def _detect_vad_segments(self, audio_chunk: np.ndarray, current_time: float):
        """使用 VAD 模型检测语音端点"""
        try:
            res = self.vad_model.generate(
                input=audio_chunk,
                cache=self.cache,
                is_final=False,
                chunk_size=VAD_CHUNK_SIZE,
            )
            
            if res and len(res) > 0:
                segments = res[0].get("value", [])
                
                # VAD 输出格式：
                # [[beg1, end1], [beg2, end2], .., [begN, endN]]：检测到语音
                # [[beg, -1]]：只检测到起始点
                # [[-1, end]]：只检测到结束点
                # []：没有检测到起始点和结束点
                # 输出结果单位为毫秒，从起始点开始的绝对时间
                
                if segments and len(segments) > 0:
                    for segment in segments:
                        beg_ms, end_ms = segment
                        
                        # 转换为秒
                        beg_sec = beg_ms / 1000.0 if beg_ms >= 0 else -1
                        end_sec = end_ms / 1000.0 if end_ms >= 0 else -1
                        
                        # 计算相对于录音开始的时间
                        if self.recording_start_time is not None:
                            relative_time = current_time - self.recording_start_time
                            
                            # 处理起始点
                            if beg_sec >= 0:
                                abs_beg = relative_time + beg_sec
                                self._add_vad_segment(abs_beg, None)
                            
                            # 处理结束点
                            if end_sec >= 0:
                                abs_end = relative_time + end_sec
                                self._add_vad_segment(None, abs_end)
                        
        except Exception as e:
            print(f"VAD 检测错误: {e}")

    def _add_vad_segment(self, beg: Optional[float], end: Optional[float]):
        """添加 VAD 端点"""
        if beg is not None:
            # 查找是否有未结束的片段
            for i, (s_beg, s_end) in enumerate(self.vad_segments):
                if s_end is None:
                    # 已有起始点，忽略新的起始点
                    return
            # 添加新的起始点
            self.vad_segments.append([beg, None])
        
        if end is not None:
            # 查找最近的未结束片段
            for i in range(len(self.vad_segments) - 1, -1, -1):
                s_beg, s_end = self.vad_segments[i]
                if s_end is None:
                    self.vad_segments[i][1] = end
                    return
            # 没有找到起始点，忽略结束点

    def _check_recording_end(self, current_time: float):
        """检查是否应该结束录音"""
        if not self.vad_segments:
            # 还没有检测到语音，继续录音
            return
        
        # 检查最后一个语音片段是否结束
        last_segment = self.vad_segments[-1]
        if last_segment[1] is not None:
            # 最后一个片段已结束，检查是否过了足够的静音时间
            silence_duration = current_time - last_segment[1]
            if silence_duration >= SILENCE_AFTER_SPEECH:
                # 检查总录音时长
                if self.recording_start_time is not None:
                    total_duration = current_time - self.recording_start_time
                    if total_duration >= 1.0:  # 至少录音 1 秒
                        print(f"语音结束，总时长: {total_duration:.2f}秒")
                        self.is_listening = False

    def _extract_and_recognize(self) -> str:
        """提取语音片段并识别"""
        if not self.audio_buffer:
            print("没有录制到音频")
            return ""
        
        # 合并所有音频块
        full_audio = np.concatenate(self.audio_buffer)
        
        # 如果没有检测到语音片段，返回空
        if not self.vad_segments:
            print("未检测到语音片段")
            return ""
        
        # 提取有效的语音片段
        speech_segments = []
        for beg, end in self.vad_segments:
            if end is not None and beg < end:
                # 转换为样本索引
                beg_sample = int(beg * SAMPLE_RATE)
                end_sample = int(end * SAMPLE_RATE)
                
                # 确保索引在有效范围内
                beg_sample = max(0, beg_sample)
                end_sample = min(len(full_audio), end_sample)
                
                if beg_sample < end_sample:
                    segment = full_audio[beg_sample:end_sample]
                    speech_segments.append(segment)
        
        if not speech_segments:
            print("没有有效的语音片段")
            return ""
        
        # 合并所有语音片段
        speech_audio = np.concatenate(speech_segments)
        
        print(f"提取到 {len(speech_segments)} 个语音片段，总时长: {len(speech_audio) / SAMPLE_RATE:.2f}秒")
        
        # 使用非流式 ASR 识别
        return self._recognize(speech_audio)

    def _recognize(self, audio: np.ndarray) -> str:
        """通过 ASR 模型识别语音（非流式）"""
        try:
            print("正在识别语音...")
            result = self.asr_model.generate(
                input=audio,
                batch_size=1
            )
            
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                print(f"✅ 识别结果: '{text}'")
                return text
            
            return ""
            
        except Exception as e:
            print(f"ASR 模型识别错误: {e}")
            return ""


# 全局实例
_recognizer = None

async def get_voice_input() -> str:
    global _recognizer
    if _recognizer is None:
        _recognizer = VoiceRecognizer()
    return await _recognizer.get_voice_input()

async def main():
    while True:
        text = await get_voice_input()
        print(f"识别结果: {text}")

if __name__ == "__main__":
    asyncio.run(main())