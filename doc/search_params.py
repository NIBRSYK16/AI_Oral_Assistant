
import re

file_path = "/home/pi/Desktop/AI_Oral_Assistant/view-source_https___www.xfyun.cn_doc_Ise_IseAPI.html#学习引擎xml输出表一.html"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    print("--- Searching for 'topic' ---")
    matches = re.finditer(r'topic', content, re.IGNORECASE)
    for i, match in enumerate(matches):
        start = match.start()
        print(f"Match {i} at {start}")
        print(re.sub(r'<[^>]+>', ' ', content[start-200:start+200]))

    print("\n--- Searching for 'free' ---")
    matches = re.finditer(r'free', content, re.IGNORECASE)
    for i, match in enumerate(matches):
        start = match.start()
        print(f"Match {i} at {start}")
        print(re.sub(r'<[^>]+>', ' ', content[start-200:start+200]))

    print("\n--- Searching for 'category' values ---")
    # Looking for table cells that might contain category definitions
    # Usually around 'read_sentence', 'read_chapter'
    matches = re.finditer(r'read_chapter', content)
    for i, match in enumerate(matches):
        if i > 2: break # strict limit
        start = match.start()
        print(f"Match {i} at {start}")
        print(re.sub(r'<[^>]+>', ' ', content[start-200:start+200]))
        
except Exception as e:
    print(e)
