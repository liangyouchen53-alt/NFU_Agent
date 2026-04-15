# 虎科資工學長 - 羅傑 你最好的麻吉

這是一個基於 RAG (檢索增強生成) 技術的本地 AI 聊天系統。身分設定為虎科大資工系最頂、最北爛的學長「羅傑」，專門回答校園生活與畢業學分等問題。

## 🛠️ 環境準備 (重要：模型安裝)

本專案使用 [Ollama](https://ollama.com/) 運行本地模型，請確保你已安裝 Ollama 並執行以下指令下載模型：

### 1. 下載大腦模型 (LLM)
這是 AI 回答問題的核心模型，特別選用台灣繁體中文優化版本。
```bash
ollama pull kenneth85/llama-3-taiwan
ollama pull nomic-embed-text