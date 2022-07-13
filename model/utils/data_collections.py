import cv2
from skeleton import Holistic
from pathlib import Path
import numpy as np

# camera
# webcam_url = "rtsp://admin:islabac123@192.168.1.49:554/Streaming/Channels/101"
video_id = 0
video_capture = cv2.VideoCapture(video_id)
VIDEO_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(VIDEO_WIDTH, VIDEO_WIDTH)

# media pipe
holistic_model = Holistic(2, 0.7, 0.5)

# data collections detail
actions = ["yes", "no"]
no_sequence = 30
sequence_length = 30
dataset_dir = Path(r"dataset")

for action in actions:
    
    for sequence in range(no_sequence):
        
        frame_num = 0
        while frame_num < sequence_length:
            
            ret, frame = video_capture.read()
            
            if not ret:
                continue
            
            # frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            image, keypoints = holistic_model.detect(frame)
            
            if not frame_num:
                cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(500)
            else:
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            
            # create new sequence dir if not exist
            sequence_dir = dataset_dir / action / str(sequence)
            if not sequence_dir.exists():
                sequence_dir.mkdir(parents=True, exist_ok=True)
            
            # save npy
            npy_path = sequence_dir / f"{frame_num}.npy"
            np.save(npy_path, keypoints)
            
            frame_num += 1
            
            key = cv2.waitKey(10)
            if key == ord("q"):
                break
            elif key == ord("s"):
                exit()

video_capture.release()
cv2.destroyAllWindows()