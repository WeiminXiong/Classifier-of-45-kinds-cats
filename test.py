from extraction import predict
from PIL import Image
import os

box_bound = predict("test.jpg")
image = Image.open("test.jpg")
if image.mode != "RGB":
    image = image.convert("RGB")

for i in range(len(box_bound)):
    cropped = image.crop(box_bound[i])
    cropped.save(os.path.join(os.getcwd(),str(i)+".jpg"))