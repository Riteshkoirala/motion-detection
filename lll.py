import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Load object detection model
mobil_net_config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

detect_model = cv2.dnn_DetectionModel(frozen_model, mobil_net_config_file)

labels_object = []
file_name = "object.txt"
with open(file_name, 'rt') as fpt:
    labels_object = fpt.read().rstrip('\n').split('\n')

detect_model.setInputSize(320, 320)
detect_model.setInputScale(1.0 / 127.5)
detect_model.setInputMean((127.5, 127.5, 127.5))
detect_model.setInputSwapRB(True)

# Motion detection variables
previous_frame = None
motion_capture_folder = "motion_captured"

# Create motion capture folder
os.makedirs(motion_capture_folder, exist_ok=True)

# Create a function to perform object and motion detection
def perform_object_and_motion_detection(video_source=None):
    global previous_frame

    video_path = None
    if video_source is not None:
        video_path = video_source
    else:
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not video_path:
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture('vvv.mp4')
    if not cap.isOpened():
        raise IOError("Cannot open video")

    output_folder = "captured_objects"
    os.makedirs(output_folder, exist_ok=True)

    desired_width = 1000
    desired_height = 600

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    canvas = tk.Canvas(root, width=desired_width, height=desired_height)
    canvas.pack()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (desired_width, desired_height))

        # Motion Detection
        if previous_frame is not None:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

            _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Apply Gaussian blur to reduce noise
            thresholded = cv2.GaussianBlur(thresholded, (5, 5), 0)

            # Apply dilation to fill gaps in motion regions
            thresholded = cv2.dilate(thresholded, None, iterations=2)

            # Find significant motion regions based on area
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Adjust the area threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Capture the motion region in the left half of the frame
                    if x < desired_width // 2:
                        motion_image = frame[y:y + h, x:x + w]
                        if not motion_image is None and motion_image.size != 0:
                            image_name = f"motion_{len(os.listdir(motion_capture_folder)) + 1}.jpg"
                            cv2.imwrite(os.path.join(motion_capture_folder, image_name), motion_image)

        previous_frame = frame.copy()

        # Object Detection
        ClassIndex, confidence, bbox = detect_model.detect(frame, confThreshold=0.55)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    object_label = labels_object[ClassInd - 1]
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, object_label, (boxes[0] + 10, boxes[1] + 48), font, fontScale=font_scale,
                                color=(0, 255, 0), thickness=3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        canvas.update()

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main GUI window
root = tk.Tk()
root.title("Object and Motion Detection GUI")

# Create buttons to open video file and open camera
open_video_button = tk.Button(root, text="Open Video File", command=perform_object_and_motion_detection)
open_video_button.pack()

open_camera_button = tk.Button(root, text="Open Camera", command=lambda: perform_object_and_motion_detection(0))
open_camera_button.pack()

# Display the GUI
root.mainloop()
