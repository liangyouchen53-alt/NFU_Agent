# school_agent.py — 校務自主 Agent
# 
# 運作方式：
# 1. 把使用者問題 + 工具清單丟給模型
# 2. 模型回傳「我要用哪個工具、帶什麼參數」
# 3. 程式執行工具，把結果再丟回給模型
# 4. 模型根據工具結果產出最終回覆
#
# 這是標準的 ReAct (Reason + Act) Agent 模式

import ollama
import json
from config import MODEL_NAME

# ── 工具定義區 ──────────────────────────────────────────
# 每個工具說明清楚「做什麼」、「什麼時候用」，讓模型自己判斷

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_school_link",
            "description": (
                "當使用者詢問任何需要前往學校系統操作的事務時使用。"
                "例如：申請在學證明、查成績、選課、繳學費、請假、申請獎學金、"
                "查看宿舍、使用學校信箱、數位學習平台等。"
                "輸入使用者想做的事，回傳對應的校務系統網址。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "使用者想辦的校務事項，例如：在學證明、成績查詢、選課"
                    }
                },
                "required": ["service"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deadline_info",
            "description": (
                "當使用者詢問截止日期、辦理期限、什麼時候要繳、幾號前要辦等問題時使用。"
                "回傳對應事項的重要期限提醒。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "要查詢期限的事項，例如：選課、退選、學費繳交"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_office_info",
            "description": (
                "當使用者詢問要去哪個辦公室、哪個單位辦理、聯絡哪個部門時使用。"
                "回傳對應行政單位的地點與聯絡方式。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "要查詢的行政單位，例如：教務處、學務處、系辦"
                    }
                },
                "required": ["department"]
            }
        }
    }
]

# ── 工具實作區 ──────────────────────────────────────────

# 校務網址查找表（比之前更完整，依語意分類）
SCHOOL_LINKS_DB = {
    "在學證明": ("在學證明申請系統", "https://sdoas.nfu.edu.tw/coe/login"),
    "在籍證明": ("在學證明申請系統", "https://sdoas.nfu.edu.tw/coe/login"),
    "成績查詢": ("校務eCare — 成績查詢", "https://ecare.nfu.edu.tw/"),
    "歷年成績": ("校務eCare — 成績查詢", "https://ecare.nfu.edu.tw/"),
    "選課":     ("校務eCare — 選課查詢", "https://ecare.nfu.edu.tw/"),
    "加退選":   ("校務eCare — 選課查詢", "https://ecare.nfu.edu.tw/"),
    "請假":     ("校務eCare — 線上請假", "https://ecare.nfu.edu.tw/"),
    "學雜費":   ("校務eCare — 學雜費減免申請", "https://ecare.nfu.edu.tw/"),
    "繳費":     ("校務eCare — 學雜費資訊", "https://ecare.nfu.edu.tw/"),
    "弱勢助學": ("校務eCare — 弱勢助學申請", "https://ecare.nfu.edu.tw/"),
    "獎學金":   ("獎學金申請資訊", "https://osa.nfu.edu.tw/zh_tw/life/scholarship/scholarship_news"),
    "信箱":     ("Mail 2000 學校信箱", "https://mail.nfu.edu.tw/cgi-bin/login?index=1"),
    "學校信箱": ("Mail 2000 學校信箱", "https://mail.nfu.edu.tw/cgi-bin/login?index=1"),
    "數位學習": ("Ulearn 數位學習平台", "https://ulearn.nfu.edu.tw/#/"),
    "ulearn":   ("Ulearn 數位學習平台", "https://ulearn.nfu.edu.tw/#/"),
    "畢業審查": ("畢業結果查詢", "https://ecare.nfu.edu.tw/"),
    "課表":     ("校務eCare — 課表查詢", "https://ecare.nfu.edu.tw/"),
    "宿舍":     ("學務處 — 宿舍申請", "https://dsa.nfu.edu.tw/files/11-1036-138.php"),
    "住宿":     ("學務處 — 宿舍申請", "https://dsa.nfu.edu.tw/files/11-1036-138.php"),
    "圖書館":   ("圖書館系統", "https://library.nfu.edu.tw/"),
    "借書":     ("圖書館系統", "https://library.nfu.edu.tw/"),
    "實習":     ("產學合作暨就業媒合", "https://cde.nfu.edu.tw/"),
}

DEADLINE_DB = {
    "選課":   "每學期開學前一週開放，確切日期請查學校行事曆。",
    "退選":   "期中考前兩週截止，逾期不受理。",
    "學費":   "每學期開學後兩週內，逾期加收滯納金。",
    "獎學金": "每學期公告時間不同，請隨時注意學務處公告。",
    "在學證明": "線上申請後約 3 個工作天可取件。",
    "畢業審查": "大四上學期結束前需完成申請。",
}

OFFICE_DB = {
    "教務處":   ("行政大樓 1 樓", "05-631-5000 分機 1200"),
    "學務處":   ("行政大樓 1 樓", "05-631-5000 分機 1300"),
    "資工系辦": ("綜三館 4 樓", "05-631-5571"),
    "圖書館":   ("圖書館大樓", "05-631-5000 分機 2100"),
    "總務處":   ("行政大樓 1 樓", "05-631-5000 分機 1100"),
}


def get_school_link(service: str) -> str:
    """查找校務系統網址"""
    # 模糊比對：找最接近的 key
    for key, (label, url) in SCHOOL_LINKS_DB.items():
        if key in service or service in key:
            return json.dumps({
                "found": True,
                "label": label,
                "url": url
            }, ensure_ascii=False)
    
    return json.dumps({
        "found": False,
        "message": f"找不到「{service}」對應的校務系統，建議直接前往 eCare 主頁查詢。",
        "url": "https://ecare.nfu.edu.tw/"
    }, ensure_ascii=False)


def get_deadline_info(topic: str) -> str:
    """查找辦理期限"""
    for key, info in DEADLINE_DB.items():
        if key in topic or topic in key:
            return json.dumps({
                "found": True,
                "topic": key,
                "deadline": info
            }, ensure_ascii=False)
    
    return json.dumps({
        "found": False,
        "message": f"「{topic}」的期限資訊未收錄，請查閱學校行事曆或直接詢問教務處。"
    }, ensure_ascii=False)


def get_office_info(department: str) -> str:
    """查找行政單位資訊"""
    for key, (location, phone) in OFFICE_DB.items():
        if key in department or department in key:
            return json.dumps({
                "found": True,
                "department": key,
                "location": location,
                "phone": phone
            }, ensure_ascii=False)
    
    return json.dumps({
        "found": False,
        "message": f"找不到「{department}」的資訊，請洽學校總機 05-631-5000。"
    }, ensure_ascii=False)


# 工具名稱 → 函式的對應表
TOOL_FUNCTIONS = {
    "get_school_link": get_school_link,
    "get_deadline_info": get_deadline_info,
    "get_office_info": get_office_info,
}


# ── Agent 主邏輯 ────────────────────────────────────────

def run_school_agent(query: str, current_time: str) -> str:
    """
    校務 Agent 主函式。
    讓模型自主決定要呼叫哪些工具，執行後整合結果回覆。
    回傳最終回覆字串。
    """

    system_prompt = f"""
你是虎科大資工系學長「羅傑 Roger」，專門協助同學處理校務問題。
當前時間：{current_time}

你有以下工具可以使用：
- get_school_link：查詢校務系統網址
- get_deadline_info：查詢辦理期限
- get_office_info：查詢行政單位地點與電話

當同學詢問校務相關事項時，請主動使用工具查詢正確資訊再回覆。
回答請使用繁體中文，語氣自然像學長在聊天，不要太正式。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query}
    ]

    # ── 第一輪：讓模型決定要用哪些工具 ──
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        options={"temperature": 0.3}  # Agent 推理用低一點的 temperature，比較穩定
    )

    assistant_msg = response["message"]
    tool_calls = assistant_msg.get("tool_calls") or []

    # 如果模型沒有呼叫任何工具，直接回傳模型的回覆
    if not tool_calls:
        return assistant_msg.get("content", "")

    # ── 第二輪：執行模型選擇的工具，把結果還給模型 ──
    messages.append({"role": "assistant", "content": assistant_msg.get("content", ""), "tool_calls": tool_calls})

    for tool_call in tool_calls:
        fn_name   = tool_call["function"]["name"]
        fn_args   = tool_call["function"]["arguments"]

        # arguments 有時候是 dict，有時候是 JSON 字串，統一處理
        if isinstance(fn_args, str):
            fn_args = json.loads(fn_args)

        # 執行對應工具
        fn = TOOL_FUNCTIONS.get(fn_name)
        if fn:
            result = fn(**fn_args)
        else:
            result = json.dumps({"error": f"找不到工具：{fn_name}"})

        # 把工具結果加入對話
        messages.append({
            "role": "tool",
            "content": result,
        })

    # ── 第三輪：讓模型根據工具結果產出最終回覆 ──
    final_response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.7}
    )

    return final_response["message"]["content"]