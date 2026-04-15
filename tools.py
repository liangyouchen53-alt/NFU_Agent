import datetime

def get_current_time():
    now = datetime.datetime.now()
    # 格式：2026年04月15日 星期三
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    date_str = now.strftime(f"%Y年%m月%d日 {weekdays[now.weekday()]}")
    
    # 也可以加上具體小時，讓 AI 更精準
    time_str = now.strftime("%H:%M")
    
    return f"{date_str} (現在時間 {time_str})"