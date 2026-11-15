"""Drop-in replacement for ``cv2.VideoCapture`` on a live RTSP stream.

A naive ``cap.read()`` loop in the same thread as the inference drifts: while
the GPU works on frame *N*, the camera keeps writing frames into the capture
buffer, so the system eventually processes the past instead of the present.
``RTSPReader`` decouples capture from processing and always hands out the most
recent frame, which keeps the end-to-end latency constant.
"""

from __future__ import annotations

import os
import threading
import time
from urllib.parse import quote

# TCP is much more reliable than UDP inside corporate networks (packet loss,
# multicast filtering) at the cost of a few milliseconds of extra latency.
_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|stimeout;5000000|rtsp_flags;prefer_tcp|max_delay;500000"
)


def build_rtsp_url(host: str, path: str, user: str = "", password: str = "") -> str:
    """Build an RTSP URL, percent-encoding credentials.

    Cameras that are provisioned with ``@``, ``:`` or ``/`` inside the password
    produce broken URLs otherwise.
    """
    if user:
        credentials = f"{quote(user, safe='')}:{quote(password, safe='')}@"
    else:
        credentials = ""
    if not path.startswith("/"):
        path = "/" + path
    return f"rtsp://{credentials}{host}{path}"


class RTSPReader:
    """Background reader that keeps only the latest frame."""

    def __init__(self, url: str, reconnect_delay: float = 2.0) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.frame_idx = 0

        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "RTSPReader":
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_OPTIONS)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def __enter__(self) -> "RTSPReader":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------
    def _open(self):
        import cv2

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open()
                if not self.cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Simple and effective recovery: drop the capture, wait 2 s and
                # reconnect. A rebooted camera or a hiccup in the router no
                # longer kills the whole pipeline.
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self.frame_idx += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self.frame_idx

    def wait_first_frame(self, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame, idx = self.read()
            if frame is not None:
                return frame, idx
            time.sleep(0.05)
        return None, -1
