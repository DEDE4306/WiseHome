from funasr import AutoModel

from config.constants import *

model = AutoModel(
    model=KWS_MODEL_PATH,  # 最好可以加载模型
    keywords="小度小度",
    device="cpu",
    disable_update=True
)

# iic/speech_charctc_kws_phone-xiaoyun_mt
# KWS_MODEL_PATH

test_wav = "D:/Project/WiseHome/test/output_custom_voice.wav"

res = model.generate(
    input=test_wav,
    cache={},
    output_dir="./outputs"
)

print(res)


