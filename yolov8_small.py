
import numpy
import ultralytics
import torch
from ultralytics import YOLO

#check if gpu is available
print("GPU available: ", torch.cuda.is_available())

print(torch.__version__) 

model = YOLO("yolov8s.pt")

# results = model.train(data="dataset/ITA.yolo/ITA.yaml", epochs=100)

#results = model.train(data="dataset/ITA.yolo/ITA.yaml", epochs=100, batch = 48, device = "0,1,2")  # use all GPUs available.


#results = model.train(data="dataset/ITA.yolo/ITA.yaml", epochs=100, batch = 48, device = "0")  # use only GPU 0.

results = model.train(
    data="dataset/ITA.yolo/ITA.yaml",
    epochs=300,
    patience=50,
    batch=48,
    device="0",
    imgsz=640,
    lr0=0.001,  # Critical change (from default 0.01)
    lrf=0.0001  # Adds learning rate decay
)