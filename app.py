import os
from pypdf import PdfReader
import ollama

# --- 設定區 ---
PDF_PATH = "data/graduation_rules.pdf"  # 確保你的 PDF 放在這個路徑
MODEL_NAME = "llama3"

def extract_text_from_pdf(file_path):
    """讀取 PDF 所有頁面的文字"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案：{file_path}，請確認 data 資料夾路徑。")
    
    print(f"正在讀取檔案：{file_path}...")
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def ask_ai_assistant(question, context):
    """
    透過 Ollama 調用 Llama 3 進行 RAG 問答
    加入了 System Prompt 強制 AI 保持專業與繁體中文
    """
    system_prompt = """
    你是一位專業的虎科大教務處助理，負責回答學生關於畢業門檻、課程學分等問題。
    請遵守以下規範：
    1. 必須使用『繁體中文』回答問題。
    2. 答案必須完全根據提供的【背景資訊】。
    3. 如果背景資訊中沒有相關內容，請回答：『抱歉，我手邊的文件中沒有關於此問題的紀錄。』
    4. 採用條列式回答，讓內容清晰易懂。
    """
    
    # 限制 Context 長度（Llama 3 8B 本地運行建議控制在 4000 字內，避免記憶體爆炸）
    truncated_context = context[:4000] 
    
    user_message = f"【背景資訊】：\n{truncated_context}\n\n【學生問題】：\n{question}"
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ],
        options={'temperature': 0.2} # 降低隨機性，讓回答更精準
    )
    return response['message']['content']

def main():
    try:
        # 1. 讀取資料
        raw_text = extract_text_from_pdf(PDF_PATH)
        print("PDF 讀取完成！")

        # 2. 進入問答迴圈
        print("-" * 30)
        print("AI 助理已就緒，請輸入您的問題（輸入 'exit' 退出）")
        
        while True:
            query = input("\n您的問題：")
            if query.lower() == 'exit':
                break
            
            print("AI 思考中...")
            answer = ask_ai_assistant(query, raw_text)
            
            print("-" * 30)
            print(f"助教回答：\n{answer}")
            print("-" * 30)

    except Exception as e:
        print(f"系統錯誤：{e}")

if __name__ == "__main__":
    main()