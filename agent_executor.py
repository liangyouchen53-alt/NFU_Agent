# agent_executor.py

from tools import get_current_time
from agent_tools import (
    get_school_service_link_logic,
    get_deadline_info_logic,
    get_office_info_logic
)

def make_tool_executor(vector_db):
    def _retrieve(query: str, k: int, year_filter: str = None) -> list[str]:
        if vector_db is None:
            return []
        retriever = vector_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        if year_filter:
            docs = [d for d in docs if d.metadata.get("year") == year_filter]
        return [d.page_content for d in docs]

    def search_graduation_rules(query: str, year: str) -> str:
        chunks = _retrieve(query, k=20, year_filter=year)
        if not chunks:
            return f"⚠️ 查無 {year} 學年度規定。請告訴使用者沒查到，並詢問他確定是 {year} 年入學的嗎？"
        return "\n\n".join(chunks)

    def search_professors(query: str) -> str:
        chunks = _retrieve(query, k=8, year_filter="professors")
        if not chunks:
            return "⚠️ 找不到符合的教授資料，請向使用者推薦直接去系辦詢問。"
        return "\n\n".join(chunks)

    def search_calendar(query: str) -> str:
        chunks = _retrieve(query, k=10, year_filter="all")
        if not chunks:
            return "⚠️ 找不到相關校曆資料。"
        return "\n\n".join(chunks)

    def search_roger_lore(query: str) -> str:
        chunks = _retrieve(query, k=6, year_filter="roger")
        if not chunks:
            return "⚠️ 沒有相關設定資料。"
        return "\n\n".join(chunks)

    return {
        "search_graduation_rules": lambda args: search_graduation_rules(
            query=args.get("query", ""), year=args.get("year", "114")
        ),
        "search_professors": lambda args: search_professors(
            query=args.get("query", "")
        ),
        "search_calendar": lambda args: search_calendar(
            query=args.get("query", "")
        ),
        "search_roger_lore": lambda args: search_roger_lore(
            query=args.get("query", "")
        ),
        "get_school_service_link": lambda args: get_school_service_link_logic(
            query=args.get("query", "")
        ),
        "get_deadline_info": lambda args: get_deadline_info_logic(
            topic=args.get("topic", "")
        ),
        "get_office_info": lambda args: get_office_info_logic(
            department=args.get("department", "")
        )
    }