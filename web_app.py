import streamlit as st
from vector_engine import initialize_vector_db
from brain import ask_ai
from tools import get_current_time
import os

# --- 1. 頁面設定 ---
st.set_page_config(page_title="虎科資工學長 - 羅傑", page_icon="💻", layout="wide")

# 強制讓對話框撐滿寬度的 CSS (如果你還是想要全螢幕效果的話)
st.markdown("""
    <style>
    .block-container {
        max-width: 100% !important;
        padding: 2rem 5rem !important;
    }
    .stChatMessage {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ✅ 修改這裡：對應你的 JPG 檔名
ROGER_AVATAR_PATH = "08740240-ca23-11ed-b8fe-8486d07069f5.jpg" 

# 檢查檔案是否存在，否則退回到機器人圖標
ROGER_AVATAR = ROGER_AVATAR_PATH if os.path.exists(ROGER_AVATAR_PATH) else "🤖"
USER_AVATAR = "🧑‍💻"

# --- 2. 初始化 Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_retriever" not in st.session_state:
    with st.spinner("📦 羅傑正在原神啟動..."):
        st.session_state.db_retriever = initialize_vector_db()

# --- 3. 側邊欄 UI ---
with st.sidebar:
    st.title("💻 系統資訊")
    
    # 檢查頭貼路徑
    if not os.path.exists(ROGER_AVATAR_PATH):
        st.error(f"⚠️ 找不到圖片：{ROGER_AVATAR_PATH}")
    else:
        st.success("✅ 羅傑頭貼已就緒")
        st.image(ROGER_AVATAR_PATH, width=100) # 在側邊欄預覽一下
    
    st.info(f"📅 當前系統時間：\n{get_current_time()}")
    
    user_year = st.selectbox("📂 你的入學學年度:", ["111", "112", "113", "114"], index=3)
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 4. 主畫面對話邏輯 ---
st.title("💬 虎科資工學長 - 羅傑")

for msg in st.session_state.messages:
    current_avatar = ROGER_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=current_avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("想問什麼？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ROGER_AVATAR):
        # ✅ 讀取中會顯示這段話，回答內不要再重複
        with st.spinner("傑寶等等，羅傑來給你找答案..."):
            res = ask_ai(
                prompt, 
                st.session_state.db_retriever, 
                st.session_state.messages[:-1], 
                user_year=user_year
            )
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})