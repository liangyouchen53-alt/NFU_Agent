# brain.py
import ollama
from config import MODEL_NAME
from tools import get_current_time, web_search_optimized

def ask_ai(query, vector_db):
    current_time_info = get_current_time()
    
    # 1. 檢索 PDF
    pdf_context = ""
    if vector_db:
        results = vector_db.similarity_search(query, k=3)
        pdf_context = "\n\n".join([doc.page_content for doc in results])
    
    # 2. 判斷是否搜尋
    web_context = ""
    time_keywords = ["今天", "現在", "禮拜幾", "星期幾", "日期"]
    if any(word in query for word in time_keywords):
        web_context = f"【系統校時】：現在確切時間為 {current_time_info}。"
    elif len(pdf_context.strip()) < 50:
        print("🔍 助教正在上網打聽...")
        web_context = web_search_optimized(query)

    # 3. 組合 Prompt
    system_prompt = f"""
    你現在是虎科大專業助教「麻吉」。
    【絕對事實】：現在是 {current_time_info}。
    【PDF資料】：{pdf_context}
    【網路情報】：{web_context}
    
    請優先信任系統時間與 PDF。若使用網路資訊請註明。語氣要像台灣大學生。
    """
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ],
        options={'temperature': 0.3}
    )
    return response['message']['content']