# from typing import Union
import mediapipe as mp
from abc import ABC, abstractclassmethod
import cv2
import numpy as np

class __MediaPipe(ABC):
    def __init__(self, 
                 model_complexity: int=1,
                 min_detection_confidence: float=0.5,
                 min_tracking_confidence: float=0.5) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.update_model()
        
    @property
    def model_complexity(self):
        return self.__model_complexity
    
    @model_complexity.setter
    def model_complexity(self, model_complexity: int):
        assert 0 <= model_complexity <= 2, ValueError("invalid model_complexity range (0-2)")
        self.__model_complexity = model_complexity
        # self.update_model()
        
    @property
    def min_detection_confidence(self):
        return self.__min_detection_confidence
    
    @min_detection_confidence.setter
    def min_detection_confidence(self, min_detection_confidence: float):
        assert 0.0 <= min_detection_confidence <= 1.0, ValueError("invalid min_detection_confidence range (0-1)")
        self.__min_detection_confidence = min_detection_confidence
        # self.update_model()
        
    @property
    def min_tracking_confidence(self):
        return self.__min_tracking_confidence
    
    @min_tracking_confidence.setter
    def min_tracking_confidence(self, min_tracking_confidence: float):
        assert 0.0 <= min_tracking_confidence <= 1.0, ValueError("invalid min_tracking_confidence range (0-1)")
        self.__min_tracking_confidence = min_tracking_confidence
        # self.update_model()
    
    @abstractclassmethod
    def update_model(self):
        pass
    
    @abstractclassmethod
    def extract_keypoints(self):
        pass
    
    @abstractclassmethod
    def draw_styled_landmarks(self):
        pass
    
    def __str__(self):
        variables = vars(self)
        info = [f"{name.split('__')[-1]}: {value}" for name, value in variables.items() if not name.startswith("drawing")]
        return "\n".join(info)


class Holistic(__MediaPipe):
    def __init__(self, 
                 model_complexity: int,
                 min_detection_confidence: float,
                 min_tracking_confidence: float):
        self.mp_holistic = mp.solutions.holistic
        super().__init__(model_complexity,
                         min_detection_confidence,
                         min_tracking_confidence)
    
    def update_model(self):
        self.model = self.mp_holistic.Holistic(model_complexity=self.model_complexity,
                                               min_detection_confidence=self.min_detection_confidence,
                                               min_tracking_confidence=self.min_tracking_confidence)
    
    def detect(self, image: np.ndarray):
        """_summary_

        Args:
            image (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(image, np.ndarray), TypeError(f"image should be numpy array")
        
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        keypoints = self.extract_keypoints(results)
        
        self.draw_styled_landmarks(image, results)
        
        return image, keypoints
    
    def draw_styled_landmarks(self, 
                              image: np.ndarray, 
                              results):
        """_summary_

        Args:
            image (np.ndarray): _description_
            results (_type_): _description_
        """
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, 
                                       results.face_landmarks, 
                                       self.mp_holistic.FACEMESH_TESSELATION, 
                                       self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                       self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                       ) 
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, 
                                       results.pose_landmarks, 
                                       self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                       self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                       ) 
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, 
                                       results.left_hand_landmarks,
                                       self.mp_holistic.HAND_CONNECTIONS, 
                                       self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                       self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                       ) 
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(image, 
                                       results.right_hand_landmarks, 
                                       self.mp_holistic.HAND_CONNECTIONS, 
                                       self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                       self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                       ) 

    def extract_keypoints(self, results) -> np.ndarray:
        """_summary_

        Args:
            results (_type_): _description_

        Returns:
            np.array: _description_
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
        

