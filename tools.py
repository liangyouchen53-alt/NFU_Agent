# tools.py
import datetime
from duckduckgo_search import DDGS

def get_current_time():
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y年%m月%d日 {weekdays[now.weekday()]}")

def web_search_optimized(query):
    now = datetime.datetime.now()
    refined_query = f"虎尾科技大學 {query} {now.year}年 公告"
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(refined_query, max_results=5)
            clean_results = [f"來源: {r['href']}\n內容: {r['body']}" for r in raw_results if len(r['body']) > 40]
            return "\n\n".join(clean_results) if clean_results else "網路查無即時資料。"
    except Exception as e:
        return f"搜尋暫時遇到障礙：{e}"