from extraction import predict
from classification import classify
import cv2
import numpy as np

class Bounding_box:
    def __init__(self, l, u, r, d):
        self.rectangle = (l, u, r, d)
        self.clazz = []

    def in_box(self, x, y):
        l, u, r, d = self.rectangle
        return l <= x <= r and u <= y <= d


def process(path):
    rects = predict(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    boxs = []
    for rect in rects:
        box = Bounding_box(*rect)
        l, u, r, d = (round(i) for i in rect)
        iimg = cv2.resize(img[int(l):int(r), int(u):int(d)], (244, 244))
        box.clazz = classify(iimg)
        boxs.append(box)
    return boxs
