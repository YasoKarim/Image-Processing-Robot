#from roboflow import Roboflow
from ultralytics import YOLO
from roboflow import Roboflow
# Initialize Roboflow API
rf = Roboflow(api_key="g5dbPnMnBUuuxAUnwbsq")

# Specify your project and version
project = rf.workspace("yk-d9yxp").project("dataset-test-ypl47")
version = project.version(1)

# Download the dataset
dataset = version.download("yolov8")


project.version(dataset.version).deploy(model_type="yolov8", model_path=f"runs/detect/train/")
# Load the YOLOv8m model
model = YOLO("yolov8m.pt")

results = model.train(data="data.yaml", epochs=50)

print(results)

model.save("Trained_garbage_model.pt")


