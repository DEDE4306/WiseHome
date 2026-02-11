"""
语音输入模块 - 增强版
正确处理VAD模型的输出格式和状态转换
"""
import sounddevice as sd
import numpy as np
from funasr import AutoModel
import asyncio
from typing import Optional, Tuple, List
import queue
import os
from datetime import datetime
import soundfile as sf

from voice_config import *


class EnhancedVoiceInputHandler:
    """增强的语音输入处理器 - 正确处理VAD状态"""
    
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
        
        # VAD状态跟踪
        self.vad_speech_started = False  # VAD是否检测到语音开始
        self.vad_speech_ended = False    # VAD是否检测到语音结束
        
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
        """检查音量"""
        volume = np.abs(audio).mean()
        volume_db = 20 * np.log10(volume + 1e-10)
        return volume, volume_db
    
    def _parse_vad_output(self, vad_segments: List) -> Tuple[bool, bool, bool]:
        """
        解析VAD输出
        
        Args:
            vad_segments: VAD模型输出的时间段列表
            
        Returns:
            Tuple[bool, bool, bool]: (有语音, 检测到开始, 检测到结束)
            
        VAD输出格式:
        - [[beg1, end1], [beg2, end2], ...]: 完整的语音段
        - [[beg, -1]]: 只检测到起始点
        - [[-1, end]]: 只检测到结束点
        - []: 没有检测到语音
        """
        if not vad_segments or len(vad_segments) == 0:
            # 空列表：没有检测到语音
            return False, False, False
        
        has_speech = False
        has_start = False
        has_end = False
        
        for segment in vad_segments:
            if len(segment) >= 2:
                beg, end = segment[0], segment[1]
                
                if beg >= 0 and end == -1:
                    # [[beg, -1]]: 检测到起始点
                    has_speech = True
                    has_start = True
                
                elif beg >= 0 and end > beg:
                    # [[beg, end]]: 完整的语音段
                    has_speech = True
                    has_start = True
                    has_end = True
                
                elif beg == -1 and end >= 0:
                    # [[-1, end]]: 检测到结束点
                    has_speech = True
                    has_end = True
        
        return has_speech, has_start, has_end
    
    def _detect_speech_vad(self, audio: np.ndarray) -> Tuple[bool, bool, bool, Optional[list]]:
        """
        使用VAD检测语音
        
        Returns:
            Tuple[bool, bool, bool, Optional[list]]: 
                (有语音, 检测到开始, 检测到结束, VAD时间段列表)
        """
        try:
            result = self.vad_model.generate(
                input=audio,
                cache={},
                is_final=False,
                chunk_size=len(audio),
            )
            
            if not result or len(result) == 0:
                return False, False, False, []
            
            res = result[0]
            
            # 解析VAD结果
            if 'value' in res:
                vad_segments = res['value']
                has_speech, has_start, has_end = self._parse_vad_output(vad_segments)
                return has_speech, has_start, has_end, vad_segments
            
            return False, False, False, []
            
        except Exception as e:
            if DEBUG:
                print(f"⚠️ VAD错误: {e}")
            return False, False, False, []
    
    def _format_vad_segments(self, segments: List) -> str:
        """格式化VAD段用于显示"""
        if not segments:
            return "[]"
        
        parts = []
        for seg in segments:
            if len(seg) >= 2:
                beg, end = seg[0], seg[1]
                if beg == -1:
                    parts.append(f"[-, {end}]")
                elif end == -1:
                    parts.append(f"[{beg}, -]")
                else:
                    parts.append(f"[{beg}, {end}]")
        
        return ", ".join(parts)
    
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
        use_volume_threshold: bool = True
    ) -> str:
        """
        获取语音输入
        
        Args:
            timeout: 超时时间
            prompt: 提示信息
            use_volume_threshold: 是否使用音量阈值过滤
            
        Returns:
            str: 识别的文本
        """
        if prompt:
            print(prompt)
        elif DEBUG:
            print("🎤 请说话...")
        
        # 重置VAD状态
        self.vad_speech_started = False
        self.vad_speech_ended = False
        
        # 录音状态
        speech_buffer = []
        silence_chunks = 0
        total_chunks = 0
        accumulated_time = 0
        
        # 计算参数
        chunk_duration = self.vad_chunk_size / self.sr
        chunk_duration_ms = chunk_duration * 1000
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
                
                # 音量检查（可选）
                if use_volume_threshold:
                    volume, volume_db = self._check_volume(audio_chunk)
                    if volume_db < VOLUME_THRESHOLD_DB:
                        # 音量太低，跳过VAD检测
                        if self.vad_speech_started and not self.vad_speech_ended:
                            silence_chunks += 1
                            speech_buffer.append(audio_chunk)
                        continue
                
                # VAD检测
                has_speech, detected_start, detected_end, vad_segments = \
                    self._detect_speech_vad(audio_chunk)
                
                # 调试信息
                if DEBUG and vad_segments:
                    segments_str = self._format_vad_segments(vad_segments)
                    status = []
                    if detected_start:
                        status.append("START")
                    if detected_end:
                        status.append("END")
                    if status:
                        print(f"  VAD: {segments_str} [{', '.join(status)}] @ {accumulated_time:.0f}ms")
                
                # 检测到语音开始
                if detected_start and not self.vad_speech_started:
                    self.vad_speech_started = True
                    if DEBUG:
                        print(f"🔴 语音开始 (时间: {accumulated_time:.0f}ms)")
                
                # 检测到语音结束
                if detected_end:
                    self.vad_speech_ended = True
                    if DEBUG:
                        print(f"🟡 VAD检测到结束点 (时间: {accumulated_time:.0f}ms)")
                
                # 状态处理
                if self.vad_speech_started:
                    speech_buffer.append(audio_chunk)
                    
                    if has_speech:
                        # 有语音，重置静音计数
                        silence_chunks = 0
                    else:
                        # 静音
                        silence_chunks += 1
                        
                        if DEBUG and silence_chunks % 3 == 0:
                            silence_duration = silence_chunks * chunk_duration
                            print(f"  静音: {silence_duration:.1f}s / {SILENCE_DURATION}s")
                    
                    # 判断是否结束
                    if silence_chunks >= max_silence_chunks or self.vad_speech_ended:
                        if len(speech_buffer) >= min_speech_chunks:
                            speech_duration = len(speech_buffer) * chunk_duration
                            if DEBUG:
                                reason = "VAD结束" if self.vad_speech_ended else "静音超时"
                                print(f"⏹️ 语音结束 ({reason}, 时长: {speech_duration:.1f}s)")
                            break
                        else:
                            if DEBUG:
                                print("⚠️ 语音太短，继续...")
                            self.vad_speech_started = False
                            self.vad_speech_ended = False
                            speech_buffer = []
                            silence_chunks = 0
                
                await asyncio.sleep(0.01)
            
            stream.stop()
            self.is_active = False
            
            # 处理录音
            if not speech_buffer:
                return ""
            
            full_audio = np.concatenate(speech_buffer)
            
            # 保存音频
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
            import traceback
            traceback.print_exc()
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
_handler: Optional[EnhancedVoiceInputHandler] = None


def get_handler() -> EnhancedVoiceInputHandler:
    """获取处理器单例"""
    global _handler
    if _handler is None:
        _handler = EnhancedVoiceInputHandler()
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
        print("=== 测试增强版语音输入 ===\n")
        print("VAD输出格式说明:")
        print("  [[beg, end]] - 完整语音段")
        print("  [[beg, -]]   - 检测到开始")
        print("  [[-, end]]   - 检测到结束")
        print("  []           - 无语音\n")
        
        result = await get_voice_input("请说话:")
        print(f"\n最终结果: '{result}'")
    
    asyncio.run(test())