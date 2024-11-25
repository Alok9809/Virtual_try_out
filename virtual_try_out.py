import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize a buffer for smoothing landmarks
previous_positions = []

def load_clothing_image(filename):
    """Load clothing image with transparency."""
    clothing = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if clothing is None:
        raise FileNotFoundError(f"Clothing image {filename} not found.")
    return clothing

def calculate_clothing_size_and_position(landmarks, frame_shape, chest_scaling=1.0):
    """Calculate clothing size and position starting from the neck."""
    frame_width, frame_height = frame_shape[1], frame_shape[0]

    # Extract landmark positions
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate average neck position
    neck_x = int((left_shoulder[0] + right_shoulder[0]) / 2 * frame_width)
    neck_y = int((left_shoulder[1] + right_shoulder[1]) / 2 * frame_height)

    # Calculate chest width with scaling
    shoulder_width = int(np.sqrt(
        (right_shoulder[0] - left_shoulder[0]) ** 2 +
        (right_shoulder[1] - left_shoulder[1]) ** 2
    ) * frame_width)
    chest_width = int(shoulder_width * chest_scaling)

    # Proportional height of the clothing
    torso_height = int(chest_width * 1.5)

    return chest_width, torso_height, neck_x, neck_y

def smooth_landmarks(current_landmarks, buffer_size=5):
    """Smooth landmarks over time using a moving average."""
    global previous_positions
    previous_positions.append(current_landmarks)
    if len(previous_positions) > buffer_size:
        previous_positions.pop(0)
    return np.mean(previous_positions, axis=0)

def overlay_clothing(frame, clothing, landmarks):
    """Overlay clothing image on the frame starting at the neck."""
    chest_width, torso_height, neck_x, neck_y = calculate_clothing_size_and_position(landmarks, frame.shape)

    # Resize clothing image
    resized_clothing = cv2.resize(clothing, (chest_width, torso_height))

    # Calculate top-left corner of placement
    top_left_x = neck_x - (chest_width // 2)
    top_left_y = neck_y

    # Clip to ensure the overlay stays within frame bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(frame.shape[1], top_left_x + chest_width)
    bottom_right_y = min(frame.shape[0], top_left_y + torso_height)

    # Adjust dimensions for clipping
    clipped_width = bottom_right_x - top_left_x
    clipped_height = bottom_right_y - top_left_y
    resized_clothing = resized_clothing[:clipped_height, :clipped_width]

    # Handle transparency for overlaying clothing
    if resized_clothing.shape[2] == 4:  # Image has an alpha channel
        clothing_rgb = resized_clothing[:, :, :3]
        alpha_channel = resized_clothing[:, :, 3] / 255.0
    else:
        clothing_rgb = resized_clothing
        alpha_channel = np.ones((clipped_height, clipped_width), dtype=np.float32)

    # Blend clothing with the frame
    for i in range(clipped_height):
        for j in range(clipped_width):
            if alpha_channel[i, j] > 0:  # Apply where clothing is not fully transparent
                frame[top_left_y + i, top_left_x + j] = (
                    alpha_channel[i, j] * clothing_rgb[i, j] +
                    (1 - alpha_channel[i, j]) * frame[top_left_y + i, top_left_x + j]
                ).astype(np.uint8)

    return frame

def main():
    # Load clothing image
    clothing = load_clothing_image('virtual_try/clothing.png')  # Update with your clothing image path

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Initialize the pose detection model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                landmarks_array = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks])
                smoothed_landmarks = smooth_landmarks(landmarks_array)

                # Overlay clothing with smoothed landmarks
                frame = overlay_clothing(frame, clothing, smoothed_landmarks)

                # Draw pose landmarks on the frame for reference
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame with clothing overlay
            cv2.imshow('Virtual Try-On', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
