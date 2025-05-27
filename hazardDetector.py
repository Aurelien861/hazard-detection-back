import base64
import json
import time
import cv2
import torch
import numpy as np
import warnings
import time
from sqlalchemy.orm import Session
from alertModel import Alert  # modèle ORM SQLAlchemy
from db import get_db_session

from ObjectDetected import ObjectDetected
from confluent_kafka import Producer

from videoStreamBuffer import VideoStreamBuffer
warnings.filterwarnings("ignore", category=FutureWarning)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "video-alerts"

producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

def detect_hazard(objects_detected, threshold_meters, depth_map, scale):
    humans = []
    forklifts = []
    for object in objects_detected:
        if int(object.cls) == 0:  # forklift
            forklifts.append(object)
        elif int(object.cls) == 1:  # human
            humans.append(object)

    for h in humans:
        for f in forklifts:
            dist, closest_pair = min_distance_between_objects(h, f, depth_map)
            dist_m = dist * scale  # Convertir en mètres
            if dist_m < threshold_meters:
                return dist_m, closest_pair
    return None, (None, None)
       

def compute_distances(human, forklift):
    left1, top1, right1, bottom1 = human.x1, human.y1, human.x2, human.y2
    left2, top2, right2, bottom2 = forklift.x1, forklift.y1, forklift.x2, forklift.y2

    dx = max(0, max(left1, left2) - min(right1, right2))
    dy = max(0, max(top1, top2) - min(bottom1, bottom2))
    return np.sqrt(dx**2 + dy**2)

def alert_hazard(hazard):
    print("⚠️ Danger détecté ! %s m entre le chariot élévateur et l'humain." % hazard)

def detectObjects(results):
    """
    Extrait les objets détectés à partir des résultats du modèle YOLOv5.
    """
    objects_detected = []
    objects_detected_raw = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2 = [coord.item() for coord in det[:4]]
        cls = int(det[5].item())
        objects_detected_raw.append(ObjectDetected(x1, y1, x2, y2, det[4].item(), cls))

    # Filtrer les objets non contenus
    for i, obj1 in enumerate(objects_detected_raw):
        contained = False
        for j, obj2 in enumerate(objects_detected_raw):
            if i != j and obj1.is_inside(obj2):
                contained = True
                break
        if not contained:
            objects_detected.append(obj1)

    return objects_detected

def get_depth_map(frame, midas, transform):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def get_3d_points(points, depth_map):
    points_3d = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < depth_map.shape[1] and 0 <= yi < depth_map.shape[0]:
            z = float(depth_map[yi, xi])
            points_3d.append((x, y, z))
    return points_3d

def min_distance_between_objects(obj1: ObjectDetected, obj2: ObjectDetected, depth_map):
    points1 = get_3d_points(obj1.get_border_points(), depth_map)
    points2 = get_3d_points(obj2.get_border_points(), depth_map)
    
    min_dist = float("inf")
    closest_pair = (None, None)
    for p1 in points1:
        for p2 in points2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < min_dist:
                min_dist = dist
                closest_pair = (p1, p2)
    return min_dist, closest_pair

def estimate_scale_from_human(human: ObjectDetected, depth_map, real_height_m=1.8):
    x1, y1, x2, y2 = map(int, [human.x1, human.y1, human.x2, human.y2])
    height_px = abs(y2 - y1)
    if height_px == 0:
        return None

    roi = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(roi)
    if avg_depth == 0:
        return None

    # Rapport entre taille réelle et unité MiDaS
    # On suppose que la hauteur projetée sur l'image correspond à une personne debout
    # => distance projetée dans l’espace 3D ≈ hauteur réelle à cette profondeur
    scale = real_height_m / avg_depth
    return scale

def publish_and_store_alert(alert_data: dict):
    cv2.imwrite(alert_data["imageUrl"], alert_data["image"])

    # Publier dans Kafka
    producer.produce(KAFKA_TOPIC, json.dumps(alert_data).encode("utf-8"))
    producer.flush()

    # Sauvegarder dans la base de données
    db: Session = get_db_session()
    db_alert = Alert(
        camera_id=alert_data["cameraId"],
        camera_name=alert_data["cameraName"],
        timestamp=alert_data["timestamp"],
        image_url=alert_data["imageUrl"],
        distance=alert_data["distance"],
        description=alert_data["description"]
    )
    db.add(db_alert)
    db.commit()
    db.close()

def detect_forklift_and_human(camera_id, camera_name, video_path = './video3.mp4', threshold=3, conf=0.25, stop_event=None):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.conf = conf

    # Charger MiDaS
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    stream = VideoStreamBuffer(video_path)

    while not stop_event.is_set():
        ret, frame = stream.get_latest_frame()
        if not ret:
            print("Aucune frame disponible...")
            time.sleep(0.05)
            continue
        
        results = model(frame)
        objects_detected = detectObjects(results)
        humans = [o for o in objects_detected if int(o.cls) == 1]
        scale = None
        color = (255, 0, 0)
        wait_time = 1
        if humans:
            print("Humain détecté, estimation de l'échelle...")
            depht_map = get_depth_map(frame, midas, transform)
            scale = estimate_scale_from_human(humans[0], depht_map)

        if scale is not None:    
            hazard, (pt_h, pt_f) = detect_hazard(objects_detected, threshold, depht_map, scale)
            if hazard is not None:
                color = (0, 0, 255)
                alert_hazard(hazard)
                # p1 = (int(pt_h[0]), int(pt_h[1]))
                # p2 = (int(pt_f[0]), int(pt_f[1]))
                for obj in objects_detected:
                    cv2.rectangle(frame, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), color, 2)
                print(f"🚨 Alert triggered for camera {camera_id}")

                # _, buffer = cv2.imencode('.jpg', frame)
                # img_base64 = base64.b64encode(buffer).decode("utf-8")
                filename = f"{camera_id}_{int(time.time() * 1000)}.jpg"
                filepath = f"./images/{filename}"
                alert = {
                    "cameraId": camera_id, 
                    "cameraName": camera_name, 
                    "timestamp": int(time.time() * 1000),
                    "imageUrl": filepath,
                    "image": frame,
                    "distance": hazard,
                    "description": "Ouvrier trop proche d'un chariot élévateur",
                }
                publish_and_store_alert(alert)
                
        # Affichage
        # for obj in objects_detected:
        #     cv2.rectangle(frame, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), color, 2)
        # cv2.imshow("Flux RTSP", frame)
        # if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        #     break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './video3.mp4'
    threshold = 2.5
    conf = 0.25
    detect_forklift_and_human(video_path, threshold, conf)