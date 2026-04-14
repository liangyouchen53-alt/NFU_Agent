# app.py
from vector_engine import initialize_vector_db
from brain import ask_ai
import datetime

def main():
    print("🚀 系統初始化中...")
    db = initialize_vector_db()
    
    print("\n" + "="*40)
    print(f"🚀 虎科大 AI 萬事通 (模組化版) 已啟動！")
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
            print(f"❌ 發生錯誤：{e}")

if __name__ == "__main__":
    main()