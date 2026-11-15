"""Cross-platform launcher for the recorder (used by run_local.bat on Windows; works anywhere).

    python launch_local.py              # what run_local.bat runs inside .venv
    python launch_local.py --no-browser --port 8001

Same behavior as run_local.command on the Mac:
  * installs requirements into the current Python if they're missing
  * admin token kept in .admin_token, station name asked once and kept in .station
  * serves on 127.0.0.1 only (this computer only), picks a free port, opens the browser
  * keeps the computer awake while the server runs (Windows/macOS), Ctrl+C stops it
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def deps_ok() -> bool:
    code = ("import fastapi, uvicorn, yaml, multipart, numpy, cv2\n"
            "assert int(cv2.__version__.split('.')[0]) < 5 and hasattr(cv2, 'CascadeClassifier')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def ensure_deps() -> None:
    if deps_ok():
        return
    print("Installing requirements (first run only, takes a minute)...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    if r.returncode != 0 or not deps_ok():
        sys.exit("ERROR: installing requirements failed (OpenCV must be 4.x). See the messages above.")


def admin_token() -> str:
    if os.environ.get("ADMIN_TOKEN"):
        return os.environ["ADMIN_TOKEN"]
    f = ROOT / ".admin_token"
    if not f.exists() or not f.read_text().strip():
        f.write_text(secrets.token_urlsafe(18) + "\n")
        try:
            os.chmod(f, 0o600)
        except OSError:
            pass
    return f.read_text().strip()


def clean_station(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())[:8]


def station() -> str:
    if os.environ.get("MEME_STATION") is not None:
        return clean_station(os.environ["MEME_STATION"])
    f = ROOT / ".station"
    if f.exists() and f.read_text().strip():
        return clean_station(f.read_text())
    if not sys.stdin.isatty():
        return ""
    print("\nFirst run on this computer: give it a short name (letters/numbers, e.g. your initials).")
    print("New participant IDs will start with it (e.g. CA-P001) so laptops can be merged later.")
    s = clean_station(input("Station name: "))
    if not s:
        sys.exit("ERROR: a station name is needed")
    f.write_text(s + "\n")
    return s


def is_recorder(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/protocol", timeout=2) as r:
            return b"protocol_version" in r.read()
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def keep_awake():
    """Ask the OS not to sleep while recording. Returns a cleanup function."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        return lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    if sys.platform == "darwin":
        p = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
        return p.terminate
    return lambda: None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--no-browser", action="store_true", default=bool(os.environ.get("NO_BROWSER")))
    a = ap.parse_args(argv)
    os.chdir(ROOT)

    import shutil
    if not shutil.which("ffmpeg"):
        hint = "winget install Gyan.FFmpeg" if sys.platform == "win32" else "brew install ffmpeg"
        sys.exit(f"ERROR: ffmpeg not found. Install it with:  {hint}\n"
                 "then open a NEW terminal window and start the recorder again.")
    ensure_deps()
    env = dict(os.environ, ADMIN_TOKEN=admin_token(), MEME_STATION=station())

    port = a.port
    if is_recorder(port):
        print(f"The recorder is already running at http://localhost:{port}, opening it.")
        if not a.no_browser:
            webbrowser.open(f"http://localhost:{port}/")
        return 0
    for _ in range(20):
        if port_free(port):
            break
        port += 1
    else:
        sys.exit(f"ERROR: no free port found near {a.port}")

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
                               "--port", str(port), "--workers", "1", "--log-level", "warning"], env=env)
    release = lambda: None
    try:
        for _ in range(60):
            if is_recorder(port):
                break
            if server.poll() is not None:
                sys.exit("ERROR: the server stopped while starting (see the messages above)")
            time.sleep(0.5)
        else:
            sys.exit("ERROR: the server did not start within 30 s")
        release = keep_awake()
        url = f"http://localhost:{port}"
        print(f"\nRecorder running at {url}  (this computer only)")
        print(f"  Admin dashboard: {url}/admin   token: {env['ADMIN_TOKEN']}")
        print(f"  Data folder:     {os.environ.get('MEME_DATA_DIR', ROOT / 'data')}")
        print(f"  Station:         {env['MEME_STATION'] or 'none (IDs P001...)'}")
        print("  Keep this window open while recording. Ctrl+C or closing the window stops it.", flush=True)
        if not a.no_browser:
            webbrowser.open(url + "/")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
        release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
