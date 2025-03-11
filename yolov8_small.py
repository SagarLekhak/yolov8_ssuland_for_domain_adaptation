
import ultralytics
import torch
from ultralytics import YOLO

#check if gpu is available
print("GPU available: ", torch.cuda.is_available())

print(torch.__version__) 

model = YOLO("yolov8s.pt")

results = model.train(data="dataset/ITA.yolo/ITA.yaml", epochs=100)



