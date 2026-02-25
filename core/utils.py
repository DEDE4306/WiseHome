# ========== 安全内容解析 ==========
def safe_content_str(content):
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        texts = [str(item.get("text", item) or "") for item in content if
                 isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(texts).strip()
    return str(content).strip()