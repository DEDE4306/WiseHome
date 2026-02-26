import sounddevice as sd
import numpy as np
import queue
import threading
import time
from funasr import AutoModel

# ==============================
# ⚙️ 核心配置参数
# ==============================
KEYWORD = "你好小爱"
SAMPLE_RATE = 16000

# 【调整】步长保持 0.1s，保证响应速度
STEP_DURATION = 0.1
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)

# 检测窗口：2.0 秒
DETECT_WINDOW_SEC = 2.0
DETECT_WINDOW_SIZE = int(SAMPLE_RATE * DETECT_WINDOW_SEC)

# 【关键调整】双阈值策略
# 1. 预触发阈值 (PRE_THRESHOLD): 只要超过这个值，认为是“正在说”，立即触发，不等说完
#    设为 0.55 左右，平衡灵敏度和误报
PRE_THRESHOLD = 0.55

# 2. 确认阈值 (CONFIRM_THRESHOLD): 如果没达到预触发，必须达到这个值才触发 (用于说完后的情况)
CONFIRM_THRESHOLD = 0.75

# 冷却时间
COOLDOWN_SECONDS = 2.5

# 能量检测阈值 (RMS)
# 如果当前窗口声音能量太低，直接跳过推理，节省资源并避免静音误报
ENERGY_THRESHOLD = 0.01

# ==============================
# 🤖 加载模型
# ==============================
print("🔄 正在加载唤醒模型...")
model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun_mt",
    keywords=KEYWORD,
    device="cpu",
    disable_update=True,
    output_dir="./outputs"
)
print("✅ 模型加载完成")

# ==============================
# 🎤 音频队列
# ==============================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        pass
    audio = indata[:, 0].astype(np.float32).copy()
    audio_queue.put(audio)


# ==============================
# 🧠 唤醒检测线程
# ==============================
last_wakeup_time = 0


def parse_kws_result(text_str):
    if not text_str or not text_str.startswith("detected"):
        return None, 0.0
    parts = text_str.split()
    if len(parts) >= 3:
        try:
            return parts[1], float(parts[2])
        except ValueError:
            return None, 0.0
    return None, 0.0


def calculate_rms(audio_data):
    """计算音频的有效值 (RMS) 作为能量指标"""
    return np.sqrt(np.mean(audio_data ** 2))


def kws_worker():
    global last_wakeup_time
    print(f"👂 开始监听: '{KEYWORD}' (预触发:{PRE_THRESHOLD}, 确认:{CONFIRM_THRESHOLD})")

    full_buffer = np.zeros(0, dtype=np.float32)

    while True:
        try:
            chunk = audio_queue.get(timeout=1)
            full_buffer = np.concatenate([full_buffer, chunk])

            if len(full_buffer) < DETECT_WINDOW_SIZE:
                continue

            current_time = time.time()

            # 冷却检查
            if current_time - last_wakeup_time < COOLDOWN_SECONDS:
                max_len = DETECT_WINDOW_SIZE + STEP_SIZE
                if len(full_buffer) > max_len:
                    full_buffer = full_buffer[-max_len:]
                continue

            # 截取窗口
            current_window = full_buffer[-DETECT_WINDOW_SIZE:]

            # 【优化1】能量检测：如果太安静，直接跳过，避免静音误报和无效计算
            rms = calculate_rms(current_window)
            if rms < ENERGY_THRESHOLD:
                # 即使跳过也要维护 buffer
                max_len = DETECT_WINDOW_SIZE + STEP_SIZE
                if len(full_buffer) > max_len:
                    full_buffer = full_buffer[-max_len:]
                continue

            # 裁剪历史
            max_len = DETECT_WINDOW_SIZE + STEP_SIZE
            if len(full_buffer) > max_len:
                full_buffer = full_buffer[-max_len:]

            # 推理
            res = model.generate(input=current_window, cache={})

            if not res:
                continue
            print(res)

            item = res[0]
            text_raw = item.get("text", "")
            text2_raw = item.get("text2", "")

            kw1, conf1 = parse_kws_result(text_raw)
            kw2, conf2 = parse_kws_result(text2_raw)

            best_keyword = None
            best_confidence = 0.0

            # 取最高置信度
            if conf1 > conf2:
                best_keyword, best_confidence = kw1, conf1
            else:
                best_keyword, best_confidence = kw2, conf2

            # 调试日志 (仅在置信度较高时打印，避免刷屏)
            if best_confidence > 0.3:
                source = "text2" if conf2 > conf1 else "text"
                # 标记是预触发还是确认触发
                tag = "[预触发!]" if best_confidence >= PRE_THRESHOLD else "[扫描中]"
                # print(f"{tag} {best_confidence:.3f} ({source}) | RMS: {rms:.4f}")

            # 【优化2】双阈值判断逻辑
            should_wake = False

            # 情况 A: 达到预触发阈值 (通常发生在说话过程中) -> 立即唤醒!
            if best_confidence >= PRE_THRESHOLD:
                should_wake = True
                print(f"\n>>> ⚡ [预触发] {best_keyword} (置信度: {best_confidence:.3f}) - 说话中检测!")

            # 情况 B: 未达到预触发，但达到了确认阈值 (通常发生在说完后) -> 唤醒
            elif best_confidence >= CONFIRM_THRESHOLD:
                should_wake = True
                print(f"\n>>> ✅ [确认触发] {best_keyword} (置信度: {best_confidence:.3f}) - 说完后检测")

            if should_wake and best_keyword:
                last_wakeup_time = time.time()
                on_wakeup(best_keyword, best_confidence)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ 检测出错: {e}")


def on_wakeup(keyword, confidence):
    print("\n" + "=" * 40)
    print(f"🎉 唤醒成功: {keyword}")
    print(f"📊 最终置信度: {confidence:.3f}")
    print(f"⏳ 冷却 {COOLDOWN_SECONDS} 秒")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    thread = threading.Thread(target=kws_worker, daemon=True)
    thread.start()

    print("🎙️ 启动麦克风...")
    try:
        with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=STEP_SIZE,
                callback=audio_callback,
        ):
            print(f"🟢 系统就绪! 请说 '{KEYWORD}'")
            print("   (策略：说话中途置信度>0.55 立即触发，无需等说完)")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n👋 退出")
    except Exception as e:
        print(f"❌ 错误: {e}")