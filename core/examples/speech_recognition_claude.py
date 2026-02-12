import sounddevice as sd
import numpy as np
from funasr import AutoModel
import asyncio
from typing import Optional, Tuple
import queue
import os
from datetime import datetime
import soundfile as sf

from voice_config import *


class ImprovedVoiceInputHandler:
    """改进的语音输入处理器"""
    
    def __init__(self):
        self.sr = SAMPLE_RATE
        self.vad_chunk_size = VAD_CHUNK_SIZE
        
        # 加载模型
        if DEBUG:
            print("正在加载模型...")
        
        self.vad_model = AutoModel(model=VAD_MODEL, disable_pbar=True)
        self.asr_model = AutoModel(model=ASR_MODEL)
        
        if DEBUG:
            print("✅ 模型加载完成")
        
        # 状态
        self.audio_queue = queue.Queue()
        self.is_active = False
        
        # 如果需要保存音频，创建目录
        if SAVE_AUDIO:
            os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)
    
    def _audio_callback(self, indata, frames, time, status):
        """音频流回调"""
        if status and DEBUG:
            print(f"⚠️ 音频状态: {status}")
        
        audio_chunk = indata[:, 0].copy()
        self.audio_queue.put(audio_chunk)
    
    def _check_volume(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        检查音量
        
        Returns:
            Tuple[float, float]: (音量均值, 音量dB)
        """
        volume = np.abs(audio).mean()
        volume_db = 20 * np.log10(volume + 1e-10)
        return volume, volume_db
    
    def _detect_speech_vad(self, audio: np.ndarray) -> Tuple[bool, Optional[list]]:
        """
        使用VAD检测语音
        
        Returns:
            Tuple[bool, Optional[list]]: (是否检测到语音, VAD时间段列表)
            VAD时间段格式: 
            - [[beg1, end1], [beg2, end2], ...]: 完整的语音段
            - [[beg, -1]]: 只检测到起始点
            - [[-1, end]]: 只检测到结束点
            - []: 没有检测到语音
        """
        try:
            result = self.vad_model.generate(
                input=audio,
                cache={},
                is_final=False,
                chunk_size=len(audio),
            )
            
            if not result or len(result) == 0:
                return False, []
            
            res = result[0]
            
            # 解析VAD结果
            if 'value' in res:
                vad_segments = res['value']
                
                if not vad_segments or len(vad_segments) == 0:
                    # 空列表：没有检测到语音
                    return False, []
                
                # 检查是否有有效的语音段
                has_speech = False
                for segment in vad_segments:
                    if len(segment) >= 2:
                        beg, end = segment[0], segment[1]
                        
                        # [[beg, -1]]: 检测到起始点
                        if beg >= 0 and end == -1:
                            has_speech = True
                            break
                        
                        # [[beg, end]]: 完整的语音段
                        if beg >= 0 and end > beg:
                            has_speech = True
                            break
                        
                        # [[-1, end]]: 检测到结束点（说明之前有语音）
                        if beg == -1 and end >= 0:
                            has_speech = True
                            break
                
                return has_speech, vad_segments
            
            return False, []
            
        except Exception as e:
            if DEBUG:
                print(f"⚠️ VAD错误: {e}")
            return False, []
    
    def _detect_speech_combined(self, audio: np.ndarray) -> Tuple[bool, Optional[list]]:
        """
        综合检测：VAD + 音量
        
        Returns:
            Tuple[bool, Optional[list]]: (是否检测到语音, VAD时间段列表)
        """
        # 音量检测
        volume, volume_db = self._check_volume(audio)
        
        if volume_db < VOLUME_THRESHOLD_DB:
            return False, []  # 音量太小
        
        # VAD检测
        return self._detect_speech_vad(audio)
    
    def _recognize(self, audio: np.ndarray) -> str:
        """识别语音"""
        try:
            result = self.asr_model.generate(
                input=audio,
                batch_size=1
            )
            
            if result and len(result) > 0 and 'text' in result[0]:
                return result[0]['text'].strip()
            
            return ""
            
        except Exception as e:
            if DEBUG:
                print(f"❌ ASR错误: {e}")
            return ""
    
    def _save_audio(self, audio: np.ndarray, prefix: str = "voice"):
        """保存音频到文件"""
        if not SAVE_AUDIO:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"
            filepath = os.path.join(AUDIO_SAVE_PATH, filename)
            sf.write(filepath, audio, self.sr)
            
            if DEBUG:
                print(f"💾 音频已保存: {filepath}")
        
        except Exception as e:
            if DEBUG:
                print(f"⚠️ 保存音频失败: {e}")
    
    async def get_voice_input(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        prompt: Optional[str] = None,
        use_volume_detection: bool = True
    ) -> str:
        """
        获取语音输入
        
        Args:
            timeout: 超时时间
            prompt: 提示信息
            use_volume_detection: 是否使用音量检测辅助VAD
            
        Returns:
            str: 识别的文本
        """
        if prompt:
            print(prompt)
        elif DEBUG:
            print("🎤 请说话...")
        
        # 状态重置
        speech_detected = False
        silence_chunks = 0
        speech_buffer = []
        total_chunks = 0
        accumulated_time = 0  # 累计时间（毫秒）
        
        # 计算参数
        chunk_duration = self.vad_chunk_size / self.sr
        chunk_duration_ms = chunk_duration * 1000  # 转换为毫秒
        max_silence_chunks = int(SILENCE_DURATION / chunk_duration)
        min_speech_chunks = int(MIN_SPEECH_DURATION / chunk_duration)
        max_total_chunks = int(MAX_RECORDING_DURATION / chunk_duration)
        
        # 启动音频流
        stream = sd.InputStream(
            samplerate=self.sr,
            channels=CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=self.vad_chunk_size
        )
        
        try:
            stream.start()
            self.is_active = True
            start_time = asyncio.get_event_loop().time()
            
            while True:
                # 超时检查
                if asyncio.get_event_loop().time() - start_time > timeout:
                    if DEBUG:
                        print("⏱️ 超时")
                    return ""
                
                # 录音时长检查
                if total_chunks >= max_total_chunks:
                    if DEBUG:
                        print("⏱️ 达到最大录音时长")
                    break
                
                # 获取音频
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                
                total_chunks += 1
                accumulated_time += chunk_duration_ms
                
                # 检测语音
                if use_volume_detection:
                    has_speech, vad_segments = self._detect_speech_combined(audio_chunk)
                else:
                    has_speech, vad_segments = self._detect_speech_vad(audio_chunk)
                
                # 调试信息：显示VAD检测结果
                if DEBUG and vad_segments:
                    if has_speech:
                        # 格式化显示VAD段
                        segments_str = ", ".join([f"[{s[0]}, {s[1]}]" for s in vad_segments])
                        print(f"  VAD检测: {segments_str} (累计时间: {accumulated_time:.0f}ms)")
                
                if has_speech:
                    if not speech_detected:
                        if DEBUG:
                            print(f"🔴 语音开始 (时间: {accumulated_time:.0f}ms)")
                        speech_detected = True
                    
                    silence_chunks = 0
                    speech_buffer.append(audio_chunk)
                    
                elif speech_detected:
                    silence_chunks += 1
                    speech_buffer.append(audio_chunk)
                    
                    # 显示静音计数
                    if DEBUG and silence_chunks % 3 == 0:  # 每3个chunk显示一次
                        silence_duration = silence_chunks * chunk_duration
                        print(f"  静音: {silence_duration:.1f}s / {SILENCE_DURATION}s")
                    
                    if silence_chunks >= max_silence_chunks:
                        if len(speech_buffer) >= min_speech_chunks:
                            speech_duration = len(speech_buffer) * chunk_duration
                            if DEBUG:
                                print(f"⏹️ 语音结束 (时长: {speech_duration:.1f}s)")
                            break
                        else:
                            if DEBUG:
                                print("⚠️ 语音太短，继续...")
                            speech_detected = False
                            speech_buffer = []
                            silence_chunks = 0
                
                await asyncio.sleep(0.01)
            
            stream.stop()
            self.is_active = False
            
            # 处理录音
            if not speech_buffer:
                return ""
            
            full_audio = np.concatenate(speech_buffer)
            
            # 保存音频(如果启用)
            self._save_audio(full_audio)
            
            # 识别
            if DEBUG:
                print("🔄 正在识别...")
            
            text = self._recognize(full_audio)
            
            if text and DEBUG:
                print(f"✅ 识别: {text}")
            elif DEBUG:
                print("❌ 未识别到内容")
            
            return text
            
        except Exception as e:
            if DEBUG:
                print(f"❌ 错误: {e}")
            return ""
            
        finally:
            if stream.active:
                stream.stop()
            stream.close()
            self.is_active = False
            
            # 清空队列
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break


# 全局实例
_handler: Optional[ImprovedVoiceInputHandler] = None


def get_handler() -> ImprovedVoiceInputHandler:
    """获取处理器单例"""
    global _handler
    if _handler is None:
        _handler = ImprovedVoiceInputHandler()
    return _handler


async def get_voice_input(
    timeout: float = DEFAULT_TIMEOUT,
    prompt: Optional[str] = None
) -> str:
    """
    便捷函数：获取语音输入
    
    Example:
        >>> text = await get_voice_input("请说出指令:")
        >>> print(f"识别到: {text}")
    """
    handler = get_handler()
    return await handler.get_voice_input(timeout=timeout, prompt=prompt)


if __name__ == "__main__":
    async def test():
        print("=== 测试语音输入 ===\n")
        
        result = await get_voice_input("请说话:")
        print(f"\n结果: '{result}'")
    
    asyncio.run(test())