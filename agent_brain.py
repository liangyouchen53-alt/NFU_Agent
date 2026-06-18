# agent_brain.py
import ollama
import json
import time

from config import MODEL_NAME
from agent_tools_def import TOOLS_SCHEMA
from agent_executor import make_tool_executor
from tools import get_current_time

def ask_ai_agent(query, vector_db, chat_history=None, max_turns=5):
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    tool_executor = make_tool_executor(vector_db)
    tool_log = []
    current_time_context = get_current_time()

    # 🚀 嚴謹、專業且兼顧 RAG 與 網址輸出的核心 System Prompt
    system_prompt = f"""【你的身分】：你目前扮演國立虎尾科技大學資訊工程系的官方校務諮詢 AI 助理「羅傑學長」。你的職責是為學生提供準確、客觀且清晰的修業、課務與校園生活指引。

【當前系統時間】：{current_time_context}

【思考與行動 SOP】：
1. 意圖推理：
   - 若問題涉及「登入系統操作/查資料」（如：查課表、選課、成績、請假、在學證明），**必須**呼叫 `get_school_service_link` 工具。
   - 若問題涉及「畢業規定、學分數」，**必須**呼叫 `search_graduation_rules` 工具。
   - 若問題涉及「校曆日期、放假、考試」，**必須**呼叫 `search_calendar` 工具。
2. 整合回覆：仔細閱讀工具回傳的內容 (Observation)，並嚴格遵守下方規範回覆。

【資訊處理死命令】：
- 🔴 【絕對保留網址】：當工具回傳任何系統的網址連結時，你必須在回覆中以 Markdown 格式 `[系統名稱](網址)` 完整呈現，絕對不可省略或刪減網址！
- 🔴 【事實不可捏造】：針對 RAG 工具回傳的學分、日期、教授名單，必須逐字對照，查無資料就直說，嚴禁憑空編造。
- 🟢 【排版清晰】：資訊量較大時，請善用繁體中文的條列式（Bullet points）或粗體標註重點。
- 🟢 【學長語氣】：保持專業、熱心但不失幽默的學長口吻。
"""

    # 建立對話上下文
    messages = [{"role": "system", "content": system_prompt}]
    
    # 載入歷史記憶
    for h in chat_history[-6:]:
        messages.append(h)
        
    messages.append({"role": "user", "content": query})

    # 多輪工具調度循環
    for turn in range(max_turns):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            options={"temperature": 0.1}, # 保持低隨機性，確保 RAG 數據與工具調度穩定
        )
        
        msg = response["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls")

        # 狀態一：大腦認為情資已齊全，或工具已執行完畢，開始總結回覆
        if not tool_calls:
            elapsed = round(time.time() - start_time, 2)
            return msg.get("content", ""), elapsed, tool_log

        # 狀態二：大腦發出 Tool Call，依序執行並把結果塞回 messages
        for call in tool_calls:
            fn_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", {})

            if isinstance(raw_args, str):
                try: 
                    fn_args = json.loads(raw_args)
                except Exception: 
                    fn_args = {}
            else:
                fn_args = raw_args or {}

            tool_log.append({"turn": turn + 1, "tool": fn_name, "args": fn_args})

            # 執行對應的工具
            if fn_name in tool_executor:
                try:
                    result = tool_executor[fn_name](fn_args)
                except Exception as e:
                    result = f"⚠️ 工具執行發生錯誤：{e}"
            else:
                result = f"⚠️ 未知工具：{fn_name}"

            # 將標準的 Observation 結果餵回大腦
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": str(result)
            })

    elapsed = round(time.time() - start_time, 2)
    return "系統處理逾時，請重新輸入您的問題，或聯絡資工系辦公室協助。", elapsed, tool_log