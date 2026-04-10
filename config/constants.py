from pathlib import Path

REACT_OUTPUT = True
USING_SPEECH_REC = True

SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 300
CHUNK_SIZE = int(CHUNK_SIZE_MS * SAMPLE_RATE / 1000)

CHANNELS = 1

# 需要修改模型存放路径，默认存放在项目根目录下的 model 文件夹
MODEL_DIR = Path(__file__).parent.parent / "model"

# 本地存放的模型文件夹名
VAD_MODEL = "fsmn-vad"
ASR_MODEL = "SenseVoiceSmall"
KWS_MODEL = "speech_charctc_kws_phone-xiaoyun"
TTS_MODEL = "Qwen3-TTS-12Hz-1.7B-CustomVoice"

# 需要将模型下载到本地
VAD_MODEL_PATH = MODEL_DIR / VAD_MODEL
ASR_MODEL_PATH = MODEL_DIR / ASR_MODEL
KWS_MODEL_PATH = MODEL_DIR / KWS_MODEL
TTS_MODEL_PATH = MODEL_DIR / TTS_MODEL

THREAD_ID = "persistent_user_session"

KEYWORD = "你好小康"

# MongoDB 连接配置
MONGODB_URI = "mongodb://localhost:27017/"  # MongoDB 服务器地址
DB_NAME = "wisehome_db"  # MongoDB 数据库名称

# 集合名称
CHAT_COLLECTION_NAME = "chat_logs"  # 聊天日志集合名称