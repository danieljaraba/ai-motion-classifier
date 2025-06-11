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
            
            print(f"Features extracted: {len(features)}")
            
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
        """Extract features matching the training data format"""
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
        
        try:
            ls = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            
            torso_inclination = np.arctan2(
                (rs.y + ls.y) - (rh.y + lh.y),
                (rs.x + ls.x) - (rh.x + lh.x)
            )
            features.append(torso_inclination)
        except:
            features.append(0.0)
        
        try:
            rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            rk = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            ra = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
            
            right_knee_angle = np.arctan2(
                ra.y - rk.y, ra.x - rk.x
            ) - np.arctan2(
                rh.y - rk.y, rh.x - rk.x
            )
            features.append(right_knee_angle)
        except:
            features.append(0.0)
        
        try:
            lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            lk = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            la = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            
            left_knee_angle = np.arctan2(
                la.y - lk.y, la.x - lk.x
            ) - np.arctan2(
                lh.y - lk.y, lh.x - lk.x
            )
            features.append(left_knee_angle)
        except:
            features.append(0.0)
        
        try:
            ls = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_width = np.sqrt((rs.x - ls.x)**2 + (rs.y - ls.y)**2)
            features.append(shoulder_width)
        except:
            features.append(0.0)
        
        try:
            lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            hip_width = np.sqrt((rh.x - lh.x)**2 + (rh.y - lh.y)**2)
            features.append(hip_width)
        except:
            features.append(0.0)
        
        try:
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            com_height = nose.y - (lh.y + rh.y) / 2
            features.append(com_height)
        except:
            features.append(0.0)
        
        while len(features) < 54:
            features.append(0.0)
        
        if len(features) > 54:
            features = features[:54]
        
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