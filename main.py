import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the YOLO network
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each output
    for out in outs:
        for detection in out:
            # Get the class ID and confidence of the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter only detections with confidence greater than threshold
            if confidence > 0.5:
                # Get bounding box parameters (center_x, center_y, width, height)
                center_x = int(detection[0] * width)  # Convert to pixel coordinates
                center_y = int(detection[1] * height)  # Convert to pixel coordinates
                w = int(detection[2] * width)  # Convert width to pixel coordinates
                h = int(detection[3] * height)  # Convert height to pixel coordinates

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append results to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0) if label == 'person' else (255, 0, 0)  # Green for person, blue for car
            
            # Increase thickness of the bounding box
            thickness = 3
            
            # Increase font size and thickness for the text
            font_scale = 1.2
            font_thickness = 2

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Put text with increased font size and thickness
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    # Show the output frame with bounding boxes
    cv2.imshow("Video Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
