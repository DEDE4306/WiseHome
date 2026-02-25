from pathlib import Path

REACT_OUTPUT = True
USING_SPEECH_REC = False

SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 300
CHUNK_SIZE = int(CHUNK_SIZE_MS * SAMPLE_RATE / 1000)

CHANNELS = 1

# 需要修改模型存放路径
MODEL_DIR = Path("D:\Project\WiseHome\model")

# 本地存放的模型文件夹名
VAD_MODEL = "fsmn-vad"
ASR_MODEL = "SenseVoiceSmall"
KWS_MODEL = "speech_charctc_kws_phone-xiaoyun_mt"

# 需要将模型下载到本地
VAD_MODEL_PATH = MODEL_DIR / VAD_MODEL
ASR_MODEL_PATH = MODEL_DIR / ASR_MODEL
KWS_MODEL_PATH = MODEL_DIR / KWS_MODEL

KEYWORD = "你好小爱"