import cv2
import numpy as np
from playsound import playsound
import threading
import time

# Flag to check if alarm is playing
is_alarm_playing = False


# Function to play sound
def play_alarm():
    global is_alarm_playing
    is_alarm_playing = True
    playsound('warning.mp3')  # Play the alarm sound
    is_alarm_playing = False  # Reset flag when done


# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red fire color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Orange fire color range
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Yellow fir      e color range
    lower_yellow = np.array([25, 150, 30])  # Reduce saturation and value for a darker yellow
    upper_yellow = np.array([35, 255, 250])  # Cap value at 200 for less brightness

    # Create masks for red, orange, and yellow ranges
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_red, mask_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)

    # Find contours of the detected fire areas
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red bounding box around each detected fire region
    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Only consider larger areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

    # Count non-zero pixels in the mask (indicating fire presence)
    fire_pixels = cv2.countNonZero(combined_mask)

    # Threshold for triggering alarm
    if fire_pixels > 5000 and not is_alarm_playing:  # Only play if alarm is not already playing
        # print("Fire detected! Triggering alarm...")
        threading.Thread(target=play_alarm).start()

    # Display the original feed with the bounding boxes
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release capture and close windows
cap.release()
cv2.destroyAllWindows()