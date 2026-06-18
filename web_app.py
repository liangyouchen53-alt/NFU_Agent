# web_app.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL, DB_PATH
from agent_brain import ask_ai_agent
from tools import get_current_time

@st.cache_resource
def load_my_vector_db():
    """
    快取 FAISS 索引，防止每次 Rerun 重複讀取硬碟，修復 NameError 缺陷。
    """
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL, 
        base_url="http://127.0.0.1:11434"
    )
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"找不到本地向量庫路徑: '{DB_PATH}'，請先執行 `vector_engine.py` 建立索引。")
        
    vector_db = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_db

try:
    db = load_my_vector_db()
except Exception as e:
    st.error(f"❌ 向量資料庫（DB）載入失敗！")
    st.code(str(e))
    db = None


st.set_page_config(page_title="虎科大資工系 AI 萬事通", page_icon="🎓", layout="centered")

current_time_str = get_current_time()
st.title("🎓 虎科大資工系 AI 萬事通")
st.caption(f"📅 系統時間：{current_time_str}")

with st.sidebar:
    st.header("⚙️ 系統設定")
    user_year = st.selectbox("請選擇你的入學學年度：", ["111", "112", "113", "114"], index=3)
    st.markdown("---")
    st.markdown("### 💡 羅傑學長叮嚀")
    st.info("\有問題快問，學長等一下還要趕著去排隊買大吉祥香豆腐！")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "學弟妹好！我是資工系辦助教「羅傑學長」，選課、畢業門檻、還是要找哪個教授，問我就對了！"
        }
    ]

if "chat_history_clean" not in st.session_state:
    st.session_state.chat_history_clean = []

for msg in st.session_state.messages:
    avatar_icon = "😎" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.write(msg["content"])

if prompt := st.chat_input("有什麼選課或畢業規定要問羅傑學長？"):
    if db is None:
        st.error("⚠️ 系統未就緒：向量資料庫未成功定義或載入。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="😎"):
        with st.spinner("學長正在翻箱倒櫃幫你查資料..."):
            ans, elapsed, tool_log = ask_ai_agent(
                query=prompt,
                vector_db=db,
                chat_history=st.session_state.chat_history_clean
            )
            
            st.write(ans)
            st.caption(f"⏱️ 思考耗時: {elapsed} 秒")
            
            if tool_log:
                with st.expander("⚙️ 查看羅傑學長的自主推理工具鏈 (Tool Log)"):
                    for log in tool_log:
                        st.markdown(f"**第 {log['turn'] + 1} 輪決策** ── 呼叫手腳工具：`{log['tool']}`")
                        st.json(log['args'])

    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.session_state.chat_history_clean.append({"role": "user", "content": prompt})
    st.session_state.chat_history_clean.append({"role": "assistant", "content": ans})