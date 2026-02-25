import json

model = {
    "text": "hello",
    "mood": "sad"
}

print(model)
print(model["text"])
print(model["mood"])

# 把一个符合 JSON 格式的字符串（str），转换成 Python 的字典（dict）、列表（list）等数据结构
hello = json.loads('{"text": "hello", "mood": "sad"}')
print(hello)
print(hello["text"])
print(hello["mood"])
print(hello.get("text"))
print(hello.get("mood"))
