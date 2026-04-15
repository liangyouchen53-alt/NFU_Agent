import ollama
from config import MODEL_NAME
from tools import get_current_time

def ask_ai(query, vector_db, chat_history, user_year="114"):
    current_time_info = get_current_time()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) if vector_db else None
    
    is_roger_lore = any(k in query.lower() for k in ["住哪", "生日", "事蹟", "黑歷史", "羅晟原", "公館路"])
    is_school_info = any(k in query for k in ["學分", "畢業", "門檻", "必修", "選修", "英文", "證照"])
    is_date_query = any(k in query for k in ["幾號", "日期", "什麼時候", "考試"])

    context_list = []
    if retriever:
        if is_roger_lore:
            res = retriever.invoke(query, filter={"year": "roger"})
            context_list.extend([doc.page_content for doc in res])
        if is_school_info or (not is_roger_lore and not is_date_query):
            res = retriever.invoke(query, filter={"year": user_year})
            context_list.extend([doc.page_content for doc in res])
        if is_date_query:
            res = retriever.invoke(query, filter={"year": "all"})
            context_list.extend([doc.page_content for doc in res])

    final_context = "\n\n".join(context_list)

    system_prompt = f"""
    【你的身分】：虎科大資工系最北爛的學長「羅傑 Roger」。
    【當前時間】：{current_time_info}
    
    【絕對指令】：
    1. 嚴禁編造數據！回答必須完全參考下方的資料。若資料說 132 學分，絕對不能說 128。
    2. 你的回答直接進入主題，不要再說「傑寶等等...」，因為讀取畫面已經顯示過了。
    3. 保持北爛口氣，絕對不准道歉。常用詞：妥當啦、哭啊、55555。

    【參考資料】：
    {final_context if final_context else "（沒資料，叫他自己去餐餐自由配啦）"}
    """
    
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(chat_history[-3:]) 
    messages.append({'role': 'user', 'content': query})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={'temperature': 0.7} 
    )
    
    return response['message']['content']