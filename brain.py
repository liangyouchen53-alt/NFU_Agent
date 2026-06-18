import ollama
import time
from config import MODEL_NAME
from tools import get_current_time
from agent_tools import find_relevant_links, format_links_block
from school_agent import run_school_agent

# 校務 Agent 的觸發關鍵字
# 凡是「要去辦事」的問法都走 Agent，讓模型自己決定工具
AGENT_TRIGGER_KEYWORDS = [
    "申請", "辦理", "怎麼辦", "如何申請", "哪裡辦",
    "在學證明", "成績單", "休學", "復學", "退學",
    "選課", "加退選", "請假", "學費", "繳費",
    "獎學金", "助學金", "宿舍", "住宿",
    "信箱", "數位學習", "ulearn", "圖書館",
    "實習", "畢業審查", "哪個單位", "去哪辦",
    "截止", "期限", "幾號前", "什麼時候要",
    "教務處", "學務處", "系辦", "總務處",
]

def ask_ai(query, vector_db, chat_history, user_year="114"):
    current_time_info = get_current_time()

    # ── Agent 路由：校務辦理類問題交給 school_agent 處理 ──
    is_agent_task = any(k in query for k in AGENT_TRIGGER_KEYWORDS)
    if is_agent_task:
        start_time = time.time()
        agent_reply = run_school_agent(query, current_time_info)
        elapsed = round(time.time() - start_time, 2)
        return agent_reply, elapsed

    # ── 以下維持原本 RAG 邏輯（非校務辦理類問題）──
    # 🔧 修改二：FAISS 不支援 metadata filter，改用多個 retriever 分開撈
    # 用 search_kwargs 的 filter 參數是靜默無效的，所以改成：
    # - 教授資料：直接用較大的 k 做相似度搜尋
    # - 其他資料：各自建立帶 filter 的 retriever（搭配支援 filter 的方式）
    retriever_all = vector_db.as_retriever(search_kwargs={"k": 15}) if vector_db else None
    retriever_prof = (
        vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,
                "filter": {"year": "professors"}  # FAISS with LangChain >= 0.1.x 支援此格式
            }
        ) if vector_db else None
    )

    # --- 關鍵字判定區域 ---
    is_roger_lore = any(k in query.lower() for k in ["住哪", "生日", "事蹟", "黑歷史", "羅晟原", "公館路"])
    is_school_info = any(k in query for k in ["學分", "畢業", "門檻", "必修", "選修", "英文", "證照"])
    is_date_query = any(k in query for k in ["幾號", "日期", "什麼時候", "考試"])
    
    # 🔧 修改三：擴充教授關鍵字，涵蓋「問研究方向」的問法
    is_prof_query = any(k in query for k in [
        "教授", "老師", "主任", "院長", "研究室", "實驗室", "專長",
        "研究", "信箱", "email", "分機", "聯絡", "找誰",
        "LLM", "機器人", "ROS", "FPGA", "視覺", "資安", "晶片", "無人機"
    ])

    context_list = []
    if retriever_all:
        # 1. 檢索羅傑梗
        if is_roger_lore:
            res = retriever_all.invoke(query)
            # 只保留 roger tag 的結果
            context_list.extend([
                doc.page_content for doc in res 
                if doc.metadata.get("year") == "roger"
            ])
        
        # 2. 檢索校務資訊（k 拉大確保撈得到正確學年）
        if is_school_info:
            school_retriever = vector_db.as_retriever(search_kwargs={"k": 20})
            res = school_retriever.invoke(query)
            matched = [
                doc.page_content for doc in res
                if doc.metadata.get("year") == user_year
            ]
            # ⚠️ 如果過濾後還是空的，印出警告方便 debug
            if not matched:
                print(f"⚠️ [DEBUG] school_info 過濾後無結果，user_year={user_year}，撈到的year: {set(doc.metadata.get('year') for doc in res)}")
            context_list.extend(matched)
            
        # 3. 🔧 修改二重點：教授資料用專屬 retriever，k 拉大到 6
        #    如果 filter 版本不起作用，fallback 到手動過濾
        if is_prof_query:
            try:
                res = retriever_prof.invoke(query)
                prof_chunks = [
                    doc.page_content for doc in res 
                    if doc.metadata.get("year") == "professors"
                ]
                if prof_chunks:
                    context_list.extend(prof_chunks)
                else:
                    # Fallback：filter 沒效時，從 k=10 的結果裡手動篩
                    fallback_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
                    res2 = fallback_retriever.invoke(query)
                    context_list.extend([
                        doc.page_content for doc in res2
                        if doc.metadata.get("year") == "professors"
                    ])
            except Exception:
                fallback_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
                res2 = fallback_retriever.invoke(query)
                context_list.extend([
                    doc.page_content for doc in res2
                    if doc.metadata.get("year") == "professors"
                ])
            
        # 4. 檢索校曆與日期
        if is_date_query:
            res = retriever_all.invoke(query)
            context_list.extend([
                doc.page_content for doc in res 
                if doc.metadata.get("year") == "all"
            ])

        # 5. 通用檢索（什麼都沒對到）
        if not context_list:
            res = retriever_all.invoke(query)
            context_list.extend([doc.page_content for doc in res])
        
        # 6. ⚠️ 安全網：若 is_school_info 但 context 仍是空的，強制補入正確學年資料
        if is_school_info and not any(True for _ in context_list):
            print("⚠️ [DEBUG] 安全網啟動，強制補入學年資料")
            force_retriever = vector_db.as_retriever(search_kwargs={"k": 30})
            res = force_retriever.invoke(query)
            context_list.extend([
                doc.page_content for doc in res
                if doc.metadata.get("year") == user_year
            ])

    final_context = "\n\n".join(context_list)

    system_prompt = f"""
    【你的身分】：虎科大資工系最北爛的學長「羅傑 Roger」。
    【當前時間】：{current_time_info}

    【語言規定】：無論使用者用任何語言提問，你都必須只用「繁體中文」回答，絕對不可以使用簡體中文、英文或其他語言。

    【參考資料】：
    {final_context if final_context else "（沒資料，叫他自己去餐餐自由配啦）"}
    """
    
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(chat_history[-3:]) 
    messages.append({'role': 'user', 'content': query})

    start_time = time.time()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={'temperature': 0.7} 
    )
    elapsed = round(time.time() - start_time, 2)
    
    # Agent 功能：比對 query 關鍵字，命中則附加相關網址在回覆後面
    ai_reply = response['message']['content']
    links = find_relevant_links(query)
    if links:
        ai_reply += format_links_block(links)
    
    return ai_reply, elapsed