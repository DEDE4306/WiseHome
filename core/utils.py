def safe_content_str(content):
    """安全字符串解析"""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        texts = [str(item.get("text", item) or "") for item in content if
                 isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(texts).strip()
    return str(content).strip()