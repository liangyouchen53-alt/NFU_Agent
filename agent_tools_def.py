# agent_tools_def.py

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_graduation_rules",
            "description": "【觸發時機】：當使用者詢問資工系『學分門檻』、『必修/選修科目』、『畢業規定』時呼叫。\n【RAG 優化關鍵】：傳入的 query 必須是『去背的純關鍵字/科目名』（例如：將「大一上要修什麼必修」轉為「第一學年 上學期 必修」；將「畢業要多少學分」轉為「最低畢業總學分」），嚴禁帶入語氣詞或整句對話！",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "精煉後的檢索關鍵字，例如：'最低畢業總學分'、'專業必修'、'院必修'、'微積分'"
                    },
                    "year": {
                        "type": "string",
                        "enum": ["111", "112", "113", "114"],
                        "description": "使用者的入學學年度。若對話無提及，請預設帶入 '114'。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_professors",
            "description": "【觸發時機】：詢問資工系『教授專長』、『研究領域』、『實驗室位置』、『聯絡分機』或職務時呼叫。\n【RAG 優化關鍵】：query 請直接帶入『教授姓名』或『專長關鍵字』（如：'鄭錦聰'、'LLM'、'網頁設計'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "教授姓名或核心技術領域關鍵字"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_calendar",
            "description": "【觸發時機】：詢問虎科大『行事曆重要日程』、『放假日期』、『考試週』、『寒暑假開始』時呼叫。\n【RAG 優化關鍵】：query 請轉換為時間或活動關鍵字（如：'期中考試'、'寒假'、'開學'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "日程或活動核心關鍵字"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_school_service_link",
            "description": "【觸發時機】：當使用者表達『想要前往、查詢、登入』學校的特定平台（如：數位學習平台、圖書館、校園信箱）或個人校務業務（如：查課表、看成績、辦理請假、查學雜費）時呼叫。\n【核心任務】：必須精確識別核心主體（如：輸入 '圖書館' 或 '數位學習'），絕對不可將所有校園平台一概簡化為 eCare。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "請輸入提取出的核心系統關鍵字，例如：'數位學習'、'圖書館'、'信箱'、'成績'、'課表'、'請假'。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deadline_info",
            "description": "【觸發時機】：當使用者詢問『截止日期』、『辦理期限』、『什麼時候要繳』時呼叫。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "要查期限的事項，如『選課』、『學費』"
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
            "description": "【觸發時機】：當使用者詢問『要去哪個處室辦』、『電話幾號』、『在哪棟樓』時呼叫。",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "行政單位名稱，如『教務處』、『系辦』"
                    }
                },
                "required": ["department"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_roger_lore",
            "description": "查詢羅傑這個虛擬角色的背景設定與黑歷史。僅限閒聊時使用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]