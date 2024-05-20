'''
from ultralytics import YOLO
from PIL import Image
#from cv2 import cv2
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8m.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')
#results = model.predict(source="0",show = True)
'''
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
#model = YOLO("best.pt")
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8m.pt')

#model = YOLO('yolov8n.yaml')

# Initialize the webcam
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict on the frame
    results = model([frame])

    # Get bounding boxes from the results
    for result in results:
        boxes = result.boxes

        # Draw bounding boxes on the frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
camera.release()
cv2.destroyAllWindows() 