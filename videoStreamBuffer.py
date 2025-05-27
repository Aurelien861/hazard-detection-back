import cv2
import threading
import time

class VideoStreamBuffer:
    def __init__(self, source, reconnect_delay=5, max_fps=30):
        """
        source : chemin vidéo local ou URL RTSP
        reconnect_delay : délai entre les tentatives de reconnexion RTSP
        max_fps : délai de lecture pour limiter la charge CPU (par défaut : 30 fps)
        """
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.delay = 1 / max_fps
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()

        self.cap = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"🔁 Tentative de connexion à la source : {self.source}")
                if not self._connect():
                    print("❌ Échec de connexion. Nouvelle tentative dans quelques secondes...")
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Lecture échouée. Reconnexion prévue...")
                self.cap.release()
                time.sleep(self.reconnect_delay)
                continue

            with self.lock:
                self.ret = True
                self.frame = frame

            time.sleep(self.delay)

    def get_latest_frame(self):
        with self.lock:
            if self.frame is None or not self.ret:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.thread.join()
