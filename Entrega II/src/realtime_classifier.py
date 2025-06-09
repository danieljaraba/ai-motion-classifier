import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time
from utils import draw_posture_analysis

class ActivityClassifier:
    def __init__(self, model_path='models/svm_model.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/scaler.pkl')
        self.pca = joblib.load('models/pca.pkl')
        self.le = joblib.load('models/label_encoder.pkl')
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.buffer = deque(maxlen=5)
        
    def process_frame(self, frame):
        """Procesa un frame y devuelve predicción"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            features = self._extract_features(results.pose_landmarks.landmark)
            scaled = self.scaler.transform([features])
            reduced = self.pca.transform(scaled)
            
            proba = self.model.predict_proba(reduced)[0]
            self.buffer.append(proba)
            avg_proba = np.mean(self.buffer, axis=0)
            
            pred_class = np.argmax(avg_proba)
            confidence = avg_proba[pred_class]
            activity = self.le.inverse_transform([pred_class])[0]
            
            return activity, confidence, results.pose_landmarks
        return None, None, None
    
    def _extract_features(self, landmarks):
        features = []
        
        JOINTS_TO_TRACK = [
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_HIP',
            'RIGHT_HIP',
            'LEFT_KNEE',
            'RIGHT_KNEE',
            'LEFT_ANKLE',
            'RIGHT_ANKLE',
            'LEFT_WRIST',
            'RIGHT_WRIST',
            'NOSE',
            'LEFT_FOOT_INDEX'
        ]
        
        for joint in JOINTS_TO_TRACK:
            try:
                lm = landmarks[mp.solutions.pose.PoseLandmark[joint].value]
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        expected_features = 72
        
        
        if len(features) != expected_features:
            raise ValueError(f"Se esperaban {expected_features} features, se obtuvieron {len(features)}")
        
        return features

def main():
    classifier = ActivityClassifier()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        activity, confidence, landmarks = classifier.process_frame(frame)
        
        if landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
            cv2.putText(frame, f"{activity} ({confidence:.2f})", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            draw_posture_analysis(frame, landmarks)
        
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Activity Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()