import cv2
import supervision as sv
from ultralytics import YOLO

# Load your trained YOLOv10 model
model = YOLO('best-2.pt')  # Replace 'best-2.pt' with the actual path to your model file

# Define class names (adjust if needed)
class_names = ["base", "max", "min", "tip", "background"]

# Initialize video capture from webcam (usually index 0)
cap = cv2.VideoCapture(0) 

# Create annotators for visualization
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    result = model(frame, conf=0.25)[0]  # Adjust confidence threshold as needed
    detections = sv.Detections.from_ultralytics(result)

    # Annotate the frame with detections
    frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections)

    # Display the annotated frame
    cv2.imshow("Webcam Gauge Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
