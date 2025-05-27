import cv2
import torch
import numpy as np

class ObjectDetected():
    def __init__(self, x1, y1, x2, y2, conf, cls):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.cls = cls

    def compute_center(self):
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        return (center_x, center_y)    
    
    def is_inside(self, other):
        return (self.x1 >= other.x1 and self.y1 >= other.y1 and
                self.x2 <= other.x2 and self.y2 <= other.y2)

    def __str__(self):
        return f"ObjectDetected(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, conf={self.conf}, cls={self.cls})"

def detect_hazard(objects_detected: ObjectDetected, threshold):
    humans = []
    forklifts = []
    for object in objects_detected:        
        if int(object.cls) == 0:  #  0 = forklift
            forklifts.append(object)
        elif int(object.cls) == 1:  # 1 = human
            humans.append(object)

    print("Humains détectés :", humans)
    print("Chariots élévateurs détectés :", forklifts)

    for h in humans:
        for f in forklifts:
            dist = compute_distances(h, f)
            if dist < threshold:
                return dist

def compute_distances(human, forklift):
    left1, top1, right1, bottom1 = human.x1, human.y1, human.x2, human.y2
    left2, top2, right2, bottom2 = forklift.x1, forklift.y1, forklift.x2, forklift.y2

    dx = max(0, max(left1, left2) - min(right1, right2))
    dy = max(0, max(top1, top2) - min(bottom1, bottom2))
    return np.sqrt(dx**2 + dy**2)

def alert_hazard(hazard):
    print("⚠️ Danger détecté ! %s pixels entre le chariot élévateur et l'humain." % hazard)

def get_latest_frame(cap, is_stream=False):
    if not is_stream:
        ret, frame = cap.read()
        return frame if ret else None

    # Pour un flux live (RTSP)
    # Vide les frames en attente et lit la plus récente
    frame = None
    while True:
        ret, tmp = cap.read()
        if not ret:
            return None
        frame = tmp
        if not cap.grab():
            break
    return frame

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


def detect_forklift_and_human(video_path = './video1.mp4', threshold=100, conf=0.25):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.conf = conf

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : impossible d’ouvrir le flux.")
        return

    while True:
        frame = get_latest_frame(cap)
        if frame is None:
            break

        print("Traitement de la frame...")
        results = model(frame)
        objects_detected = detectObjects(results)

        hazard = detect_hazard(objects_detected, threshold)

        color = (255, 0, 0)
        wait_time = 1
        if hazard is not None:
            wait_time = 500
            color = (0, 0, 255)
            alert_hazard(hazard)

        # Affichage
        for obj in objects_detected:
            cv2.rectangle(frame, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), color, 2)
        cv2.imshow("Flux RTSP", frame)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './video1.mp4'
    threshold = 10
    conf = 0.25
    detect_forklift_and_human(video_path, threshold, conf)