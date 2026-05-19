# watch_roasts.py
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 防抖：记录上次触发时间，避免短时间内多次运行
last_run = 0
debounce_seconds = 2  # 2秒内多次触发只执行一次

class RoastFolderHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        # 只关注 roasts 文件夹内的 .xls 或 .xlsx 文件的变化
        if not event.src_path.endswith(('.xls', '.xlsx')):
            return
        global last_run
        now = time.time()
        if now - last_run < debounce_seconds:
            return
        last_run = now
        print(f"[{time.ctime()}] 检测到变化: {event.event_type} - {event.src_path}")
        # 延迟 0.5 秒，确保文件操作完成
        time.sleep(0.5)
        subprocess.run(["python", "process_data.py"])
        print(f"[{time.ctime()}] 数据处理完成")

if __name__ == "__main__":
    path = r"C:\Users\86181\Desktop\CC_MVP\roasts"
    event_handler = RoastFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"监控文件夹: {path} (包括创建、删除、修改)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()