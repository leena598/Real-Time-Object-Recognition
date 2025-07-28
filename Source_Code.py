from ultralytics import YOLO
import cv2
import time

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Choose the YOLOv8 variant: yolov8n.pt (nano), yolov8s.pt (small), etc.

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0


cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    # Capture frame-by-frame
   
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame_id += 1

    # Run YOLOv8 inference
    results = model(frame, stream=True)

    # Process and visualize results
    for result in results:
        boxes = result.boxes  # Extract bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordinates of the bounding box
            conf = round(box.conf[0].item(), 2)  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls]  # Class label

            # Draw bounding box and label
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf}", (x1, y1 - 10), font, 0.5, color, 2)

    # Calculate and display FPS
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (40, 40), font, 0.7, (0, 255, 255), 1)
    cv2.putText(frame, "Press [ESC] to exit", (40, 70), font, 0.5, (0, 255, 255), 1)

    # Display the frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        print("[Button Pressed] ///// [ESC]")
        print("[Feedback] ///// Video capturing successfully stopped")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()