import sounddevice as sd
import numpy as np
import queue
import threading
from funasr import AutoModel

# ==============================
# 配置参数
# ==============================

KEYWORD = "小云小云"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 秒
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

CONFIDENCE_THRESHOLD = 0.7

# ==============================
# 加载模型
# ==============================

print("加载唤醒模型...")

model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun_mt",
    keywords=KEYWORD,
    device="cpu"
)

print("模型加载完成")

# ==============================
# 音频队列
# ==============================

audio_queue = queue.Queue()

# ==============================
# 麦克风回调
# ==============================

def audio_callback(indata, frames, time, status):

    if status:
        print("音频状态:", status)

    audio = indata[:, 0].copy()

    audio_queue.put(audio)

# ==============================
# 唤醒检测线程
# ==============================

cache = {}

def kws_worker():

    print("开始监听唤醒词:", KEYWORD)

    buffer = np.zeros(0, dtype=np.float32)

    while True:

        chunk = audio_queue.get()

        buffer = np.concatenate([buffer, chunk])

        # 保证最小检测长度
        if len(buffer) < SAMPLE_RATE:
            continue

        # 只保留最近2秒，避免无限增长
        if len(buffer) > SAMPLE_RATE * 2:
            buffer = buffer[-SAMPLE_RATE * 2:]

        res = model.generate(
            input=buffer,
            cache=cache,
        )

        text = res[0]["text"]

        if text.startswith("detected"):

            parts = text.split()

            keyword = parts[1]
            confidence = float(parts[2])

            print(f"\n检测到唤醒词: {keyword}")
            print(f"置信度: {confidence:.3f}")

            if confidence > CONFIDENCE_THRESHOLD:
                on_wakeup(keyword, confidence)

# ==============================
# 唤醒触发事件
# ==============================

def on_wakeup(keyword, confidence):

    print("\n========== 唤醒成功 ==========")
    print("关键词:", keyword)
    print("置信度:", confidence)
    print("==============================\n")

    # 在这里触发你的 WiseHome 系统
    # 例如：
    # start_asr()
    # 或
    # turn_on_device()

# ==============================
# 启动线程
# ==============================

thread = threading.Thread(target=kws_worker, daemon=True)
thread.start()

# ==============================
# 启动麦克风
# ==============================

print("启动麦克风...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
):

    print("系统已启动，说:", KEYWORD)

    while True:
        sd.sleep(1000)
