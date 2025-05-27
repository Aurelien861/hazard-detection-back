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
    
    def get_border_points(self):
        # Points au centre de chaque côté
        points = [
            ((self.x1 + self.x2) / 2, self.y1),
            ((self.x1 + self.x2) / 2, self.y2),
            (self.x1, (self.y1 + self.y2) / 2),
            (self.x2, (self.y1 + self.y2) / 2)
        ]
        return points

    def __str__(self):
        return f"ObjectDetected(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, conf={self.conf}, cls={self.cls})"