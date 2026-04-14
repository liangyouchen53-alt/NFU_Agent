import os
import datetime
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from duckduckgo_search import DDGS 
import ollama

# --- 設定區 ---
DATA_PATH = "data/"
DB_PATH = "vector_db"
MODEL_NAME = "kenneth85/llama-3-taiwan"

def initialize_vector_db():
    """初始化向量資料庫"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("⚠️ 提示：目前 data 資料夾內沒有 PDF，僅能使用閒聊與網路搜尋。")
        return None

    print("📖 正在優化 PDF 索引與知識切片...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=MODEL_NAME),
        persist_directory=DB_PATH
    )
    return vector_db

def web_search_optimized(query):
    """優化後的搜尋功能，加入時間過濾"""
    now = datetime.datetime.now()
    refined_query = f"虎尾科技大學 {query} {now.year}年 公告"
    
    try:
        # 使用 random 避免 chrome 版本衝突警告
        with DDGS() as ddgs:
            raw_results = ddgs.text(refined_query, max_results=5)
            
            clean_results = []
            for r in raw_results:
                if len(r['body']) > 40:
                    clean_results.append(f"來源: {r['href']}\n內容: {r['body']}")
            
            return "\n\n".join(clean_results) if clean_results else "網路查無即時資料。"
    except Exception as e:
        return f"搜尋暫時遇到障礙：{e}"

def ask_ai(query, vector_db):
    """具備即時時間意識的決策大腦"""
    
    # --- 關鍵修正：注入系統即時時間 ---
    now = datetime.datetime.now()
    # 產出格式：2026年04月15日 星期三 (根據你電腦目前的正確時間)
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    current_time_info = now.strftime(f"%Y年%m月%d日 {weekdays[now.weekday()]}")
    
    pdf_context = ""
    if vector_db:
        results = vector_db.similarity_search(query, k=3)
        pdf_context = "\n\n".join([doc.page_content for doc in results])
    
    # 判斷是否需要搜尋
    web_context = ""
    time_keywords = ["今天", "現在", "禮拜幾", "星期幾", "日期"]
    news_keywords = ["最新", "公告", "新聞", "活動"]

    # 如果是問日期，直接使用系統時間，不需浪費資源搜尋
    if any(word in query for word in time_keywords):
        web_context = f"【系統校時中心】：現在確切時間為 {current_time_info}。"
    elif len(pdf_context.strip()) < 50 or any(word in query for word in news_keywords):
        print("🔍 正在進行精準網路打聽...")
        web_context = web_search_optimized(query)

    # 精準化 System Prompt，強制模型對齊時間
    system_prompt = f"""
    你現在是虎科大專業助教「麻吉」。
    
    【絕對事實 - 不可違背】：
    1. 現在的正確時間是：{current_time_info}。
    2. 如果使用者問「今天禮拜幾」或日期，請直接根據上述時間回答。
    3. 嚴禁參考搜尋結果中任何與上述時間衝突的日期。

    【資訊處理權重】：
    - 優先權 1 (最高)：系統校時資訊 (處理時間問題)
    - 優先權 2：內部文件 PDF (處理專業規則)
    - 優先權 3：網路情報 (處理即時動態)

    【資料庫內容】：{pdf_context}
    【外部參考資訊】：{web_context}
    """
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ],
        options={'temperature': 0.3} # 調低溫度，增加事實準確度
    )
    return response['message']['content']

def main():
    db = initialize_vector_db()
    
    print("\n" + "="*40)
    print(f"🚀 虎科大 AI 萬事通 (時控校準版) 已啟動！")
    print(f"當前系統時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*40)

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ['exit', 'quit', '掰掰']:
            break
        
        if not user_input.strip():
            continue

        print("AI 思考中...")
        try:
            ans = ask_ai(user_input, db)
            print(f"\n助教麻吉：\n{ans}")
        except Exception as e:
            print(f"❌ 執行出錯：{e}")

if __name__ == "__main__":
    main()