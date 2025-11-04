import cv2
from ultralytics import YOLO

# --- SETTINGS ---
# Path to your custom-trained .pt file
MODEL_PATH = "fall_yolov8n_best.pt"  

# Path to the test video you downloaded
VIDEO_PATH = "gettyimages-85131821-640_adpp.mp4" 
# -----------------

def main():
    # Load the YOLO model
    model = YOLO(MODEL_PATH)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}.")
        return

    print("Playing video and running detection. Press 'q' to quit.")

    while True:
        # Read one frame from the video
        ret, frame = cap.read()
        
        # If frame was not read (e.g., end of video), break the loop
        if not ret:
            print("End of video.")
            break

        # Run YOLOv8 inference
        results = model(frame, stream=True)

        # Draw the results on the frame
        for r in results:
            frame = r.plot() 

        # Display the annotated frame
        
        try:
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('YOLOv8 Fall Detection', display_frame)
        except Exception as e:
            print(f"Could not resize frame, displaying original. Error: {e}")
            cv2.imshow('YOLOv8 Fall Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()