# web_app.py
import streamlit as st
from vector_engine import initialize_vector_db
from brain import ask_ai
import datetime

# 網頁頁面設定
st.set_page_config(page_title="虎科 AI 萬事通", page_icon="🎓", layout="wide")

# 初始化 Session State (用來儲存聊天紀錄)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    with st.spinner("🚀 系統初始化中，正在載入 PDF 知識庫..."):
        st.session_state.db = initialize_vector_db()

# --- 側邊欄 (Sidebar) ---
with st.sidebar:
    st.title("🎓 虎科大 AI 助教")
    st.subheader("專案狀態")
    st.success("✅ 系統已啟動")
    st.info(f"📅 今日日期：{datetime.datetime.now().strftime('%Y-%m-%d')}")
    
    st.divider()
    st.markdown("""
    ### 功能說明
    - **PDF 檢索**：自動讀取 `data/` 資料夾文件。
    - **即時搜尋**：若資料庫無資料，自動聯網查詢。
    - **在地化**：使用 Llama-3-Taiwan 模型。
    """)
    
    if st.button("清空聊天紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 主介面 (Chat Interface) ---
st.title("💬 虎科大 AI 萬事通")
st.caption("我是你的助教麻吉，有什麼關於學校或生活的問題都可以問我喔！")

# 顯示歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 使用者輸入
if prompt := st.chat_input("請輸入您的問題..."):
    # 1. 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 產出 AI 回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("麻吉助教正在思考並打聽中..."):
            try:
                # 呼叫我們之前寫好的 brain.py 邏輯
                response = ask_ai(prompt, st.session_state.db)
                message_placeholder.markdown(response)
                # 儲存紀錄
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ 發生錯誤：{e}"
                st.error(error_msg)