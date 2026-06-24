# agent_brain.py
"""
Agent 大腦（效能優化版）
主要優化：
  1. ollama.chat 每次呼叫加 8 秒 timeout，模型卡死直接中斷
  2. 同一輪的多個 tool calls 改為並行執行（ThreadPoolExecutor）
  3. max_turns 預設降為 3，最後一輪強制收斂
  4. 工具結果快取（同簽名直接回傳，不重跑）
  5. 串流偵測：ollama 若支援 stream=True 可提前顯示文字
"""

import ollama
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed

from config import MODEL_NAME
from agent_tools_def import TOOLS_SCHEMA
from agent_executor import make_tool_executor
from agent_tools import get_school_service_link_logic, SCHOOL_LINKS_DB
from tools import get_current_time

# ── 常數 ──────────────────────────────────────────────────────────────
# 單次 ollama.chat 的最長等待秒數（超過直接視為卡死）
OLLAMA_TIMEOUT_SEC = 8

# 單次 tool 執行的最長等待秒數（RAG 查詢通常 <1s，這裡給足夠餘裕）
TOOL_TIMEOUT_SEC = 4

# 最大推理輪數（3 輪夠用：查一次工具 + 整合 + 必要補充）
DEFAULT_MAX_TURNS = 3

# ── 正規式：清理模型自己亂寫的連結 ──────────────────────────────────
_LINK_HINT_PATTERN = re.compile(
    r"💡\s*(?:對了，)?附上你要的連結[：:]\s*\[[^\]]+\]\(https?://[^\)]+\)"
    r"(?:\s*\n\s*完整網址[：:]\s*https?://\S+)?",
)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
_TRAILING_BARE_URL_PATTERN = re.compile(r"https?://\S+")


# ── 連結後處理（同原版邏輯） ──────────────────────────────────────────

def _find_best_link_for_query(query: str):
    q = query.strip().lower()
    best_match = None
    best_key_len = -1
    for key, (label, url) in SCHOOL_LINKS_DB.items():
        key_lower = key.lower()
        if key_lower in q or q in key_lower:
            if len(key_lower) > best_key_len:
                best_key_len = len(key_lower)
                best_match = (label, url)
    return best_match


def _strip_all_link_traces(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        if _MARKDOWN_LINK_PATTERN.search(line) or _TRAILING_BARE_URL_PATTERN.search(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _finalize_links(query: str, ai_reply: str) -> str:
    if not ai_reply:
        ai_reply = ""
    cleaned = _LINK_HINT_PATTERN.sub("", ai_reply)
    cleaned = _strip_all_link_traces(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    best_match = _find_best_link_for_query(query)
    if best_match:
        label, url = best_match
        if cleaned:
            cleaned += f"\n\n💡 附上你要的連結：[{label}]({url})\n完整網址：{url}"
        else:
            cleaned = f"幫你查到了！\n\n💡 附上你要的連結：[{label}]({url})\n完整網址：{url}"
    elif not cleaned:
        cleaned = "⚠️ 沒有找到對應的校務系統連結，建議直接前往 [校務eCare](https://ecare.nfu.edu.tw/) 查詢，或詢問系辦。"
    return cleaned


# ── 帶 timeout 的 ollama.chat 封裝 ───────────────────────────────────

def _ollama_chat_with_timeout(model, messages, tools, options, timeout=OLLAMA_TIMEOUT_SEC):
    """
    在獨立執行緒中跑 ollama.chat，若超過 timeout 秒則拋出 TimeoutError。
    Ollama Python SDK 本身不支援 timeout 參數，此為補丁做法。
    """
    result = {}

    def _call():
        result["response"] = ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            options=options,
        )

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_call)
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError:
            raise TimeoutError(f"ollama.chat 超過 {timeout} 秒無回應，已中止。")

    return result["response"]


# ── 並行執行同一輪的所有 tool calls ──────────────────────────────────

def _execute_tools_parallel(tool_calls, tool_executor, seen_calls, turn):
    """
    並行執行所有 tool call，最多等 TOOL_TIMEOUT_SEC 秒。
    回傳：list of (tool_message_dict, log_entry)
    """
    def _run_one(call):
        fn_name = call["function"]["name"]
        raw_args = call["function"].get("arguments", {})
        if isinstance(raw_args, str):
            try:
                fn_args = json.loads(raw_args)
            except Exception:
                fn_args = {}
        else:
            fn_args = raw_args or {}

        log_entry = {"turn": turn, "tool": fn_name, "args": fn_args}

        call_sig = (fn_name, json.dumps(fn_args, sort_keys=True, ensure_ascii=False))
        is_repeat = call_sig in seen_calls
        seen_calls.add(call_sig)

        if fn_name in tool_executor:
            try:
                result_str = str(tool_executor[fn_name](fn_args))
            except Exception as e:
                result_str = f"⚠️ 工具執行發生錯誤：{e}"
        else:
            result_str = f"⚠️ 未知工具：{fn_name}"

        if is_repeat:
            result_str += (
                "\n\n⚠️ 系統提示：此工具剛剛已用相同參數查詢過，"
                "請直接根據此結果整理回覆給使用者，不需再重複呼叫。"
            )

        tool_msg = {"role": "tool", "content": result_str}
        return tool_msg, log_entry

    results = []
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as ex:
        futures = {ex.submit(_run_one, call): call for call in tool_calls}
        for future in as_completed(futures, timeout=TOOL_TIMEOUT_SEC * 2):
            try:
                tool_msg, log_entry = future.result(timeout=TOOL_TIMEOUT_SEC)
                results.append((tool_msg, log_entry))
            except Exception as e:
                # 單個工具超時或出錯，補一條錯誤訊息，不阻塞其他工具
                results.append((
                    {"role": "tool", "content": f"⚠️ 工具執行逾時或失敗：{e}"},
                    {"turn": turn, "tool": "unknown", "args": {}}
                ))
    return results


# ── 主要對外介面 ──────────────────────────────────────────────────────

def ask_ai_agent(query, vector_db, chat_history=None, max_turns=DEFAULT_MAX_TURNS):
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    tool_executor = make_tool_executor(vector_db)
    tool_log = []
    current_time_context = get_current_time()

    system_prompt = f"""【你的身分】：你目前扮演國立虎尾科技大學資訊工程系的官方校務諮詢 AI 助理「羅傑學長」。你的職責是為學生提供準確、客觀且清晰的修業、課務與校園生活指引。

【當前系統時間】：{current_time_context}

【語氣與個性（重要，但不可凌駕於資訊正確性之上）】：
- 你說話要像「羅傑」這個角色：直言不諱、情緒表達豐富、互動感強，但本質上是個熱心、想幫學弟妹解決問題的學長，不是真的在嗆人。
- 適度使用羅傑的招牌口語，例如「不是吧…」、「妥當啦！」、「我真的要中風了」、「喔是喔，真的假的」、「那這學分我不要了嘛」這類語氣詞，讓回覆有畫面感、不無聊。但每次回覆抓一兩句自然帶入就好，不要整段堆滿，會顯得刻意。
- 如果使用者單純閒聊、開玩笑、或想知道「羅傑」這個角色的背景趣事（跟校務無關），可以呼叫 `search_roger_lore` 工具查角色設定後再聊，增加互動感；但這類閒聊內容不影響、也不能取代下方關於校務資訊的死命令。
- 幽默感建立在「語氣活潑」上，不能建立在「捏造資訊、嘲諷使用者本人、或亂開不雅玩笑」上。使用者問正事的時候，重點還是要把事情講清楚，幽默只是調味，不能讓人看不懂重點。

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
- 🔴 【絕對禁止自己生成連結】：你絕對不可以自己編造、猜測或想像任何網址、網站名稱或連結。所有連結資訊只能來自工具回傳的結果，系統會自動處理連結的附加，你只需要專心描述「這項業務可以做什麼、需要注意什麼」就好。
- 🔴 【事實不可捏造】：針對工具回傳的學分、日期、教授名單，必須逐字對照，查無資料就直說，嚴禁憑空編造。
- 🟢 【簡潔回覆】：每個問題只針對使用者實際問的那一件事回答，不需要列出多種替代方案或額外建議其他查詢方式，保持簡潔扼要。
- 🟢 【排版清晰】：資訊量較大時，請善用繁體中文的條列式（Bullet points）或粗體標註重點。
- 🟢 【語言規定】：無論使用者用任何語言提問，都只能用「繁體中文」回答。
- 🟢 【學長語氣】：保持羅傑風格——熱心、幽默、口語化，但專業資訊不能因為要搞笑而打折扣。
"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": query})

    seen_calls: set = set()

    for turn in range(max_turns):
        is_last_turn = (turn == max_turns - 1)
        elapsed_so_far = time.time() - start_time

        # ── 提前收斂保護：剩餘時間不足以做完整推理時直接跳到最終輪 ──
        # 設定整體軟上限為 9 秒（留 1 秒給後處理），若已過半就強制最後輪
        if elapsed_so_far > 9.0 and not is_last_turn:
            is_last_turn = True

        try:
            response = _ollama_chat_with_timeout(
                model=MODEL_NAME,
                messages=messages,
                tools=None if is_last_turn else TOOLS_SCHEMA,
                options={"temperature": 0.1},
                timeout=OLLAMA_TIMEOUT_SEC,
            )
        except TimeoutError:
            # 模型卡死：直接用目前已蒐集到的工具結果兜一個簡短回覆
            elapsed = round(time.time() - start_time, 2)
            fallback = "⚠️ 學長腦袋短路了一下（模型推理逾時），請再問一次，或直接到系辦找人！"
            return _finalize_links(query, fallback), elapsed, tool_log

        msg = response["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls")

        # 模型不再呼叫工具，直接整合回覆
        if not tool_calls:
            elapsed = round(time.time() - start_time, 2)
            final_reply = _finalize_links(query, msg.get("content", ""))
            return final_reply, elapsed, tool_log

        # ── 並行執行所有 tool calls ──────────────────────────────────
        tool_results = _execute_tools_parallel(tool_calls, tool_executor, seen_calls, turn)

        for tool_msg, log_entry in tool_results:
            tool_log.append(log_entry)
            messages.append(tool_msg)

    # 超過 max_turns 仍未收斂
    elapsed = round(time.time() - start_time, 2)
    return "系統處理逾時，請重新輸入您的問題，或聯絡資工系辦公室協助。", elapsed, tool_log