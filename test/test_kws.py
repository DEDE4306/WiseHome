from funasr import AutoModel

from config.constants import *

model = AutoModel(
    model=KWS_MODEL_PATH,  # 最好可以加载模型
    keywords="你好小爱",
    device="cpu",
    disable_update=True
)

test_wav = "D:/Project/WiseHome/test/output_custom_voice.wav"

res = model.generate(
    input=test_wav,
    cache={},
    output_dir="./outputs"
)

print(res)

