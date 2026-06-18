# cli_app.py
import datetime
from vector_engine import initialize_vector_db  # 假設你原來的初始化函式叫這個名
from agent_brain import ask_ai_agent

def main():
    print("🚀 虎科大 AI 萬事通核心啟動中（正在載入 FAISS 向量庫與 8B 大腦）...")
    
    # 載入向量資料庫
    try:
        db = initialize_vector_db()
    except Exception:
        # 如果你的 vector_engine.py 裡面的函式叫其他名字，請自行調整此處
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings
        from config import EMBED_MODEL, DB_PATH
        embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    print("\n" + "="*50)
    print(f"🎓 虎科大資工系 AI 萬事通 (8B 代理自主推理版) 已就緒！")
    print(f"系統啟動時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("輸入 'exit' 或 'quit' 即可結束對話。")
    print("="*50)

    # 用來存放完整的歷史記憶
    chat_history = []

    while True:
        try:
            user_input = input("\n😎 你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n學長先溜啦！有問題再來系辦找我！")
            break

        if user_input.lower() in ['exit', 'quit', '掰掰', '再見']:
            print("\n助教羅傑：妥當啦，退訓！祝你歐趴啦！")
            break

        if not user_input:
            continue

        print("🤖 羅傑學長正在思考與調度資源...")
        
        # 🚀 全權交給 Agent 大腦進行自主推理
        ans, elapsed, tool_log = ask_ai_agent(
            query=user_input, 
            vector_db=db, 
            chat_history=chat_history
        )

        # 打印工具調度日誌（讓你在畫面上能一眼看穿 AI 在背後偷偷做了什麼事）
        if tool_log:
            print("-"*40)
            for log in tool_log:
                print(f"🛠️ [第 {log['turn']+1} 輪決策] 呼叫手腳工具: {log['tool']} -> 參數: {log['args']}")
            print("-"*40)

        print(f"\n🚬 羅傑學長：\n{ans}")
        print(f"⏱️ 思考耗時：{elapsed} 秒")

        # 🚀 更新持久記憶（一來一回結構清晰，避免 8B 模型產生記憶斷層）
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()