# agent_tools.py
"""
校務資料庫（網址、期限、處室資訊）與對應的查詢邏輯。
這裡是「真正會被執行」的程式碼，agent_tools_def.py 只是給 LLM 看的工具說明書。
"""

# ==========================================
# ── 1. 資料庫區 (Databases) ──
# ==========================================

# 🌟 校務系統與網站連結
# key 是觸發關鍵字，value 是 (顯示名稱, 網址)
SCHOOL_LINKS_DB = {
    "數位學習": ("Ulearn 數位學習平台", "https://ulearn.nfu.edu.tw/#/"),
    "ulearn": ("Ulearn 數位學習平台", "https://ulearn.nfu.edu.tw/#/"),
    "網路大學": ("Ulearn 數位學習平台", "https://ulearn.nfu.edu.tw/#/"),

    "圖書館": ("圖書館全球資訊網", "https://lib.nfu.edu.tw/"),
    "借書": ("圖書館全球資訊網", "https://lib.nfu.edu.tw/"),

    "信箱": ("Mail 2000 學校信箱", "https://mail.nfu.edu.tw/cgi-bin/login?index=1"),
    "校園信箱": ("Mail 2000 學校信箱", "https://mail.nfu.edu.tw/cgi-bin/login?index=1"),
    "校信": ("Mail 2000 學校信箱", "https://mail.nfu.edu.tw/cgi-bin/login?index=1"),

    "在學證明": ("在學證明申請系統", "https://sdoas.nfu.edu.tw/coe/login"),
    "在籍證明": ("在學證明申請系統", "https://sdoas.nfu.edu.tw/coe/login"),

    "獎學金": ("獎學金申請資訊", "https://osa.nfu.edu.tw/zh_tw/life/scholarship/scholarship_news"),

    # 校務 eCare 負責的整合型業務
    "成績": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "期中考成績": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "期末考成績": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "選課": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "加退選": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "課表": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "請假": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "學雜費": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "繳費": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "畢業審查": ("校務eCare", "https://ecare.nfu.edu.tw/"),
    "畢業結果": ("校務eCare", "https://ecare.nfu.edu.tw/"),
}

# 📅 期限與時程規定
DEADLINE_DB = {
    "選課": "初選通常在學期末，加退選在開學第一週。",
    "退選": "期中考前兩週截止，逾期不受理。",
    "學費": "開學前兩週需完成繳費，逾期加收滯納金。",
    "獎學金": "每學期公告時間不同，請隨時注意學務處公告。",
    "在學證明": "線上申請後約 3 個工作天可取件。",
    "畢業審查": "大四上學期結束前需完成申請。",
}

# 🏢 處室位置與聯絡資訊
OFFICE_DB = {
    "教務處": ("行政大樓 1 樓", "05-631-5000 分機 1200"),
    "學務處": ("行政大樓 1 樓", "05-631-5000 分機 1300"),
    "資工系辦": ("綜三館 4 樓", "05-631-5571"),
    "圖書館": ("圖書館大樓", "05-631-5000 分機 2100"),
    "總務處": ("行政大樓 1 樓", "05-631-5000 分機 1100"),
}


# ==========================================
# ── 2. 邏輯執行區 (Tool Logic Functions) ──
# ==========================================

def get_school_service_link_logic(query: str) -> str:
    """
    查找校務系統網址。
    比對策略（由嚴到寬，盡量提高命中率，因為 LLM 不一定會乖乖傳精簡關鍵字）：
      1. key 出現在 query 裡（例如 query="我要去ulearn上課" 命中 key="ulearn"）
      2. query 出現在 key 裡（例如 query="信箱" 命中 key="校園信箱"）
      3. 都沒比對到 -> 回防呆訊息，引導去 eCare 或换個說法
    """
    raw_query = query.strip()
    q = raw_query.lower()

    for key, (label, url) in SCHOOL_LINKS_DB.items():
        key_lower = key.lower()
        if key_lower in q or q in key_lower:
            return f"💡 找到系統連結：[{label}]({url})\n完整網址：{url}"

    return (
        "⚠️ 查無此特定業務的直達連結。\n"
        "- 如果是個人課表、成績、請假等教務系統操作，請前往 [校務eCare](https://ecare.nfu.edu.tw/)\n"
        "- 如果是尋找特定處室辦公室，請試著詢問學長『XX處室在哪裡』或『XX處室電話』。"
    )


def get_deadline_info_logic(topic: str) -> str:
    """查找辦理期限與時程"""
    topic = topic.strip()
    for key, info in DEADLINE_DB.items():
        if key in topic or topic in key:
            return f"💡 關於「{key}」的時程提示：{info}"

    return "⚠️ 查無此事項的明確期限，建議直接詢問相關處室或查看教務處/學務處最新公告。"


def get_office_info_logic(department: str) -> str:
    """查找處室位置與電話"""
    department = department.strip()
    for key, (location, phone) in OFFICE_DB.items():
        if key in department or department in key:
            return f"💡 【{key}】資訊：\n- 📍 位置：{location}\n- 📞 聯絡電話：{phone}"

    return "⚠️ 查無此單位的具體位置與分機。如果是系所辦公室，可以試著加上『系辦』兩個字（例如：資工系辦）。"