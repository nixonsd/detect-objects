import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Path to the test image
# image_path = "test_image.jpeg"  # Replace with your image file path
# image_path = "test_image2.jpg"  # Replace with your image file path
image_path = "test_image3.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Image at path '{image_path}' could not be loaded.")
    exit()

height, width, channels = image.shape
print(f"Loaded image with shape: {image.shape}")

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Forward pass through the YOLO network
outs = net.forward(output_layers)
print("Completed forward pass through YOLO network.")

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

            # Debugging: Print the bounding box coordinates
            print(f"Detected {classes[class_id]} - Confidence: {confidence:.2f}, Box: x={x}, y={y}, w={w}, h={h}")

            # Append results to lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(f"Indexes after Non-Maximum Suppression: {indexes}")

# Draw bounding boxes on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0) if label == 'person' else (255, 0, 0)  # Green for person, blue for car
        
        # Increase thickness of the bounding box
        thickness = 3
        
        # Increase font size and thickness for the text
        # font_scale = 3
        font_scale = 1
        # font_thickness = 8
        font_thickness = 1

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Put text with increased font size and thickness
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)


# Show the output image with bounding boxes
cv2.imshow("Image Object Detection", image)

# Wait until any key is pressed, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
