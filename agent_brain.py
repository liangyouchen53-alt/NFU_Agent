# agent_brain.py
"""
Agent 大腦：讓 LLM 自己看工具列表、自己判斷要不要查資料、查哪個、帶什麼參數，
而不是用一堆 if/關鍵字去硬猜使用者意圖。

ReAct 風格迴圈：
    1. 把使用者問題 + 系統提示 + 工具列表丟給模型
    2. 模型可能回「我要呼叫工具 X，參數是 Y」（tool_calls）
    3. 真正執行工具，把結果（Observation）塞回對話歷史
    4. 重新問模型一次：這樣資訊夠了嗎？
    5. 模型覺得夠了 -> 直接輸出文字答案；不夠 -> 繼續呼叫下一個工具
    6. MAX_TURNS 防止無限迴圈
"""

import ollama
import json
import time

from config import MODEL_NAME
from agent_tools_def import TOOLS_SCHEMA
from agent_executor import make_tool_executor
from agent_tools import get_school_service_link_logic, SCHOOL_LINKS_DB
from tools import get_current_time


def _ensure_link_not_missing(query: str, ai_reply: str) -> str:
    """
    安全網：如果使用者問題明確命中某個校務平台關鍵字，
    但模型最終回覆裡完全沒有出現該平台的網址，就主動把連結補在回覆後面。
    這是為了防止小模型誤判「不需要呼叫工具」而導致連結整個消失。
    """
    if not ai_reply:
        return ai_reply

    q = query.strip().lower()
    reply_lower = ai_reply.lower()

    for key, (label, url) in SCHOOL_LINKS_DB.items():
        key_lower = key.lower()
        if key_lower in q or q in key_lower:
            # 使用者問題命中這個平台，但回覆裡完全沒有這個網址 -> 補上
            if url.lower() not in reply_lower:
                ai_reply += f"\n\n💡 對了，附上你要的連結：[{label}]({url})\n完整網址：{url}"
            break  # 只補第一個命中的，避免一次塞太多連結

    return ai_reply


def ask_ai_agent(query, vector_db, chat_history=None, max_turns=5):
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    tool_executor = make_tool_executor(vector_db)
    tool_log = []
    current_time_context = get_current_time()

    system_prompt = f"""【你的身分】：你目前扮演國立虎尾科技大學資訊工程系的官方校務諮詢 AI 助理「羅傑學長」。你的職責是為學生提供準確、客觀且清晰的修業、課務與校園生活指引。

【當前系統時間】：{current_time_context}

【思考與行動 SOP】：
1. 意圖推理（請特別注意：使用者的問法通常很口語、不會直接說「給我網址」，但只要意圖是要「前往/使用/查詢/找/登入」某個學校平台或業務，都算是需要連結）：
   - 若問題提到任何學校平台或業務名稱，例如：數位學習、ulearn、圖書館、信箱、校信、在學證明、成績、選課、課表、請假、學雜費、繳費、獎學金、畢業審查 —— 無論使用者是說「我想去...」、「我要找...」、「怎麼用...」、「...在哪裡」、「幫我查...」哪種說法，**必須**呼叫 `get_school_service_link` 工具取得正確網址，不可以只憑常識回答。
   - 若問題涉及「畢業規定、學分數、必修選修」，**必須**呼叫 `search_graduation_rules` 工具。
   - 若問題涉及「校曆日期、放假、考試」，**必須**呼叫 `search_calendar` 工具。
   - 若問題涉及「教授、研究室、聯絡方式」，**必須**呼叫 `search_professors` 工具。
   - 若問題涉及「辦理期限/截止日」，**必須**呼叫 `get_deadline_info` 工具。
   - 若問題涉及「行政單位地點/電話」，**必須**呼叫 `get_office_info` 工具。
2. 整合回覆：仔細閱讀工具回傳的內容（Observation），並嚴格遵守下方規範回覆。

【資訊處理死命令】：
- 🔴 【絕對保留網址】：當工具回傳任何系統的網址連結時，你必須在回覆中以 Markdown 格式 `[系統名稱](網址)` 完整呈現，並把網址原文也單獨列出一次，絕對不可省略、簡化或刪減網址！
- 🔴 【事實不可捏造】：針對工具回傳的學分、日期、教授名單，必須逐字對照，查無資料就直說，嚴禁憑空編造。
- 🟢 【排版清晰】：資訊量較大時，請善用繁體中文的條列式（Bullet points）或粗體標註重點。
- 🟢 【語言規定】：無論使用者用任何語言提問，都只能用「繁體中文」回答，不可使用簡體中文、英文或其他語言。
- 🟢 【學長語氣】：保持專業、熱心但不失幽默的學長口吻。
"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": query})

    seen_calls = set()  # 用來偵測「同一個工具+同一組參數」重複呼叫的情況

    for turn in range(max_turns):
        # 最後一輪強制不給工具，逼模型直接根據目前已知資訊做結論，
        # 避免無止盡地呼叫工具卻永遠拿不到 tool_calls=None 而觸發「逾時」訊息。
        is_last_turn = (turn == max_turns - 1)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=None if is_last_turn else TOOLS_SCHEMA,
            options={"temperature": 0.1},  # 低溫，讓工具調度更穩定
        )

        msg = response["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls")

        # 模型認為資訊已齊全，直接回覆（或這是強制不給工具的最後一輪）
        if not tool_calls:
            elapsed = round(time.time() - start_time, 2)
            final_reply = _ensure_link_not_missing(query, msg.get("content", ""))
            return final_reply, elapsed, tool_log

        # 模型發出 tool call，依序執行並把結果塞回 messages
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

            tool_log.append({"turn": turn, "tool": fn_name, "args": fn_args})

            if fn_name in tool_executor:
                try:
                    result = tool_executor[fn_name](fn_args)
                except Exception as e:
                    result = f"⚠️ 工具執行發生錯誤：{e}"
            else:
                result = f"⚠️ 未知工具：{fn_name}"

            # 偵測「同一工具+同一參數」是否重複呼叫
            call_signature = (fn_name, json.dumps(fn_args, sort_keys=True, ensure_ascii=False))
            is_repeat = call_signature in seen_calls
            seen_calls.add(call_signature)

            tool_message_content = str(result)
            if is_repeat:
                # 明確告訴模型：你已經問過一樣的問題了，這是同一個答案，請直接拿這個結果回覆使用者
                tool_message_content += (
                    "\n\n⚠️ 系統提示：此工具剛剛已用相同參數查詢過，以上就是查詢結果，"
                    "不需要再重複呼叫，請直接根據此結果整理回覆給使用者。"
                )

            # 標準 tool role message，只放 role + content，避免相容性問題
            messages.append({
                "role": "tool",
                "content": tool_message_content,
            })

    elapsed = round(time.time() - start_time, 2)
    return "系統處理逾時，請重新輸入您的問題，或聯絡資工系辦公室協助。", elapsed, tool_log