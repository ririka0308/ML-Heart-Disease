import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port(start: int = 8505, end: int = 8515) -> int:
    for port in range(start, end + 1):
        if not _is_port_in_use(port):
            return port
    return start


def _clear_streamlit_cache() -> None:
    """删除所有缓存，确保页面显示最新代码"""
    home = Path.home()
    # Streamlit 缓存
    cache_dir = home / ".streamlit" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    # Python 编译缓存（__pycache__）
    root = Path(__file__).resolve().parent
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)

    # Streamlit 浏览器数据缓存
    streamlit_data = home / ".streamlit"
    if streamlit_data.exists():
        for item in streamlit_data.iterdir():
            if item.name != "config.toml" and item.is_dir():
                shutil.rmtree(item)

    print("已清除所有缓存。")


def main() -> None:
    root = Path(__file__).resolve().parent

    # 每次启动先清缓存，确保页面代码是最新的
    _clear_streamlit_cache()

    port = 8505
    if _is_port_in_use(port):
        print(f"端口 {port} 被占用，自动寻找其他端口...")
        port = _find_free_port(port + 1)
        print(f"已切换到端口 {port}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
    ]

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        time.sleep(0.5)
        print("系统已安全关闭。")


if __name__ == "__main__":
    sys.excepthook = lambda typ, val, tb: (
        print("系统已安全关闭。") if typ is KeyboardInterrupt else __import__("traceback").print_exception(typ, val, tb)
    )
    main()
