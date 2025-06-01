from random import randint
import cv2
import av
import asyncio
import threading
import time
import json
from typing import Dict
import asyncio

from fastapi import Depends, FastAPI, WebSocket, Request, HTTPException, WebSocketDisconnect
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from fastapi.middleware.cors import CORSMiddleware
from aiokafka import AIOKafkaConsumer
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from db import get_db_session
from alertModel import Alert

from hazardDetector import detect_forklift_and_human


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="images"), name="images")
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "video-alerts"
relay = MediaRelay()

# Camera manager
class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def start_camera(self, camera_id, rtsp_url, name):
        with self.lock:
            if camera_id in self.cameras:
                return
            
            stop_event = threading.Event()
            
            reader_thread = threading.Thread(target=self._reader_thread, args=(camera_id, rtsp_url), daemon=True)
            detection_thread = threading.Thread(
                target=detect_forklift_and_human,
                args=(camera_id, name),
                kwargs={
                    "video_path": rtsp_url,
                    "threshold": 2.5,
                    "conf": 0.25,
                    "stop_event": stop_event
                },
                daemon=True
            )

            self.cameras[camera_id] = {
                "reader_thread": reader_thread,
                "detect_thread": detection_thread,
                "frame": None,
                "running": True,
                "url": rtsp_url,
                "name": name,
                "stop_event": stop_event
            }
            reader_thread.start()
            detection_thread.start()

    def stop_camera(self, camera_id):
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id]["running"] = False
                self.cameras[camera_id]["stop_event"].set()
                del self.cameras[camera_id]

    def _reader_thread(self, camera_id, rtsp_url):
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return
        while self.cameras[camera_id]["running"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            with self.lock:
                self.cameras[camera_id]["frame"] = frame
            time.sleep(0.03)
        cap.release()

    def get_frame(self, camera_id):
        with self.lock:
            return self.cameras.get(camera_id, {}).get("frame")
        
    def get_active_cameras(self):
        with self.lock:
            return [
                {"id": cam_id, "url": data.get("url"), "name": data.get("name")}
                for cam_id, data in self.cameras.items()
                if data.get("running")
            ]

camera_manager = CameraManager()


# WebRTC video track
class OpenCVCameraStream(VideoStreamTrack):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = camera_manager.get_frame(self.camera_id)
        if frame is None:
            await asyncio.sleep(0.1)
            return await self.recv()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

@app.post("/start-stream/")
async def start_stream(request: Request):
    data = await request.json()
    camera_id = data["id"]
    name = data["name"]
    print(f"Starting stream for camera {camera_id} with URL {data['url']}")

    if data["url"] == "rtsp://admin:PII2025@192.168.1.100/stream":
        rtsp_url = "./video1.mp4"
    elif data["url"] == "rtsp://admin:PII2025@192.168.1.101/stream":
        rtsp_url = "./video3.mp4"
    elif data["url"] == "rtsp://admin:PII2025@192.168.1.102/stream":
        rtsp_url = "./video1-long.mp4"    
    else:
        raise HTTPException(status_code=400, detail="Invalid RTSP URL or camera unreachable")

    camera_manager.start_camera(camera_id, rtsp_url, name)

    return {"status": "stream started"}

@app.post("/stop-stream/{camera_id}")
async def stop_stream(camera_id: str):
    camera_manager.stop_camera(camera_id)
    return {"status": "stream stopped"}

@app.post("/offer/{camera_id}")
async def offer(camera_id: str, request: Request):
    params = await request.json()
    pc = RTCPeerConnection()

    @pc.on("iceconnectionstatechange")
    def on_ice_state_change():
        if pc.iceConnectionState == "failed":
            asyncio.ensure_future(pc.close())

    track = OpenCVCameraStream(camera_id)
    pc.addTrack(track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

@app.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    await websocket.accept()

    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=None,
        auto_offset_reset='latest'
    )

    await consumer.start()
    print("Kafka consumer started")

    try:
        async for msg in consumer:
            data = msg.value.decode("utf-8")

            try:
                # Optionnel : valider/transformer le message
                await websocket.send_text(data)
            except WebSocketDisconnect:
                print("Client disconnected during send")
                break
    except Exception as e:
        print(f"Erreur Kafka: {e}")
    finally:
        await consumer.stop()
        print("Kafka consumer stopped")

@app.get("/active-cameras")
async def get_active_cameras():
    cameras = camera_manager.get_active_cameras()
    return {"active_cameras": cameras} 

@app.get("/api/alerts")
def get_alerts(db: Session = Depends(get_db_session)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return [
        {
            "id": alert.id,
            "cameraId": alert.camera_id,
            "cameraName": alert.camera_name,
            "timestamp": alert.timestamp,
            "imageUrl": alert.image_url,
            "distance": alert.distance,
            "description": alert.description
        }
        for alert in alerts
    ]       
