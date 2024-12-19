from ultralytics import YOLO
import cv2
import time
import pyttsx3

# Load the YOLO model
model = YOLO("yolo_model.pt")  # Path to your trained YOLO model

# Initialize webcam
video = cv2.VideoCapture(0)

# Initialize TTS engine
engine = pyttsx3.init()

# Variables for tracking detections and timing
constructed_sentence = ""  # Stores the full detected sentence
last_detected_time = time.time()  # Tracks the last detection time
displayed_letter = ""  # Tracks the last detected letter
countdown_start = 3  # Countdown timer in seconds
current_countdown = countdown_start  # Current countdown value

print("Press 'q' to quit and hear the detected sentence.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame from BGR to RGB (YOLO expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(frame_rgb, conf=0.5, show=False)  # Set confidence threshold

    current_time = time.time()
    detected = False

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        conf = result.conf[0]                     # Confidence score
        label = result.cls[0]                     # Class label index
        class_name = model.names[int(label)]      # Get class name

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process the detected letter
        detected = True
        displayed_letter = class_name

        # Update the sentence if 3 seconds have passed since the last addition
        if current_time - last_detected_time >= countdown_start:
            constructed_sentence += displayed_letter
            last_detected_time = current_time
            current_countdown = countdown_start  # Reset countdown timer
            print(f"Detected letter: {displayed_letter}")

    # Update countdown if no detection
    if not detected:
        time_since_last = current_time - last_detected_time
        current_countdown = max(0, countdown_start - int(time_since_last))

        # Add a space if countdown reaches 0
        if time_since_last >= countdown_start:
            if constructed_sentence and constructed_sentence[-1] != " ":
                constructed_sentence += " "
                print("New word started.")
            last_detected_time = current_time
            current_countdown = countdown_start  # Reset countdown timer

    # Display the constructed sentence on the screen
    cv2.putText(frame, f"Sentence: {constructed_sentence}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the countdown on the screen
    cv2.putText(frame, f"Countdown: {current_countdown}s", (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame with detections and updates
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Output the constructed sentence
print(f"Constructed sentence: {constructed_sentence}")

# Say the sentence
engine.say(f"The constructed sentence is: {constructed_sentence}")
engine.runAndWait()
