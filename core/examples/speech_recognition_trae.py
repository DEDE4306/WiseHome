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
            disable_pbar=True
        )
        self.cache = {}
        self.result_text = ""
        self.final_result = ""
        self.is_listening = False
        
        # 语音检测参数
        self.silence_threshold = 0.005  # 静音阈值（可调整）
        self.silence_duration = 1.0     # 静音持续时间（秒）
        self.min_speech_duration = 0.3  # 最小语音时长（秒）
        self.debug_mode = False         # 调试模式
        
        # 状态变量
        self.speech_detected = False    # 是否检测到语音
        self.speech_start_time = None   # 语音开始时间
        self.silence_start_time = None  # 静音开始时间
        self.audio_buffer = []          # 音频缓冲区（用于最终识别）
        self.last_recognition_time = 0  # 上次识别时间
        
        # 异步处理队列
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.processing_stop_event = threading.Event()


    # 这句话不知道什么意思，为什么有事件循环这种让人摸不清头脑的东西
    async def get_voice_input(self) -> str:
        """启动一次语音输入，返回识别出的完整句子"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_and_recognize)

    def _record_and_recognize(self) -> str:
        """录音并识别语音"""
        self.final_result = ""
        self.is_listening = True
        
        # 重置状态
        self.speech_detected = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer = []
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
            print("🎤 请说话（说完后停顿1秒）...")
            # 等待语音结束
            while self.is_listening:
                sd.sleep(100)

        # 停止处理线程
        self.processing_stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2)

        print(f"✅ 识别完成: '{self.final_result}'")
        return self.final_result.strip()

    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数 - 只做轻量级处理"""
        if status:
            if status.input_overflow:
                # 只在第一次溢出时打印
                if not hasattr(self, '_overflow_logged'):
                    print("警告: 音频缓冲区溢出，考虑增大 blocksize 或减少处理量")
                    self._overflow_logged = True
            else:
                print(f"Audio status: {status}")

        audio_chunk = indata[:, 0]
        
        # 计算音量（RMS）
        volume = np.sqrt(np.mean(audio_chunk ** 2))
        
        # 将音频块和处理信息放入队列，异步处理
        self.processing_queue.put({
            'audio_chunk': audio_chunk,
            'volume': volume,
            'timestamp': time.currentTime
        })

    def _processing_worker(self):
        """后台处理线程 - 处理音频识别逻辑"""
        while not self.processing_stop_event.is_set():
            try:
                # 从队列获取数据，超时 0.1 秒
                data = self.processing_queue.get(timeout=0.1)
                
                audio_chunk = data['audio_chunk']
                volume = data['volume']
                current_time = data['timestamp']
                
                # 保存音频到缓冲区
                self.audio_buffer.append(audio_chunk)
                
                # 调试日志
                if self.debug_mode:
                    print(f"[DEBUG] 音量: {volume:.6f}, 语音检测: {self.speech_detected}")
                
                # 检测语音状态
                if volume > self.silence_threshold:
                    # 检测到语音
                    if not self.speech_detected:
                        self.speech_detected = True
                        self.speech_start_time = current_time
                        self.silence_start_time = None
                        if self.debug_mode:
                            print("[DEBUG] 检测到语音开始")
                    else:
                        # 语音继续，重置静音计时
                        self.silence_start_time = None
                else:
                    # 检测到静音
                    if self.speech_detected and self.silence_start_time is None:
                        self.silence_start_time = current_time
                        if self.debug_mode:
                            print("[DEBUG] 检测到静音开始")
                
                # 实时识别（可选，用于显示中间结果）
                if self.speech_detected:
                    try:
                        res = self.model.generate(
                            input=audio_chunk,
                            cache=self.cache,
                            is_final=False,
                            chunk_size=[0, 10, 5],
                            encoder_chunk_look_back=4,
                            decoder_chunk_look_back=1
                        )
                        
                        if res and len(res) > 0:
                            text = res[0].get("text", "").strip()
                            if text and text != self.result_text:
                                self.result_text = text
                                if self.debug_mode:
                                    print(f"[实时] {text}")
                    except Exception as e:
                        if self.debug_mode:
                            print(f"[DEBUG] 实时识别错误: {e}")
                
                # 判断是否应该结束
                if self.speech_detected and self.silence_start_time is not None:
                    silence_duration = current_time - self.silence_start_time
                    speech_duration = current_time - self.speech_start_time
                    
                    # 满足结束条件：静音时间足够 且 语音时间足够
                    if (silence_duration >= self.silence_duration and 
                        speech_duration >= self.min_speech_duration):
                        if self.debug_mode:
                            print(f"[DEBUG] 语音结束 - 静音时长: {silence_duration:.2f}s, 语音时长: {speech_duration:.2f}s")
                        self._finalize_recognition()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程错误: {e}")
    
    def _finalize_recognition(self):
        """完成识别，处理缓冲区中的所有音频"""
        if not self.audio_buffer:
            self.is_listening = False
            return
        
        print("🔄 正在处理完整音频...")
        
        # 合并所有音频块
        full_audio = np.concatenate(self.audio_buffer)
        
        # 使用完整的音频进行最终识别
        try:
            res = self.model.generate(
                input=full_audio,
                cache={},
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            
            if res and len(res) > 0:
                self.final_result = res[0].get("text", "").strip()
        except Exception as e:
            print(f"最终识别错误: {e}")
            self.final_result = self.result_text
        
        # 清理状态
        self.audio_buffer = []
        self.speech_detected = False
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
