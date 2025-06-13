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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.buffer = deque(maxlen=10)
        self.prev_landmarks = None
        
    def process_frame(self, frame):
        """Procesa un frame y devuelve predicción"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            features = self._extract_features(results.pose_landmarks.landmark)
            
            if len(features) != self.scaler.n_features_in_:
                print(f"Error: Se esperaban {self.scaler.n_features_in_} características, pero se obtuvieron {len(features)}")
                return None, None, None
            
            total_movement = self._calculate_total_movement(results.pose_landmarks.landmark)
            
            if total_movement < 0.01:  # Umbral de movimiento
                self.prev_landmarks = results.pose_landmarks.landmark
                return "quieto", 0.95, results.pose_landmarks
            
            scaled = self.scaler.transform([features])
            reduced = self.pca.transform(scaled)
            
            proba = self.model.predict_proba(reduced)[0]
            self.buffer.append(proba)
            
            weights = np.linspace(0.5, 1.0, len(self.buffer))
            weights = weights / weights.sum()
            avg_proba = np.average(self.buffer, axis=0, weights=weights)
            
            pred_class = np.argmax(avg_proba)
            confidence = avg_proba[pred_class]
            activity = self.le.inverse_transform([pred_class])[0]
            
            self.prev_landmarks = results.pose_landmarks.landmark
            return activity, confidence, results.pose_landmarks
        return None, None, None
    
    def _extract_features(self, landmarks):
        """Extract features matching the training data format with 87 características"""
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
        
        # Características derivadas - necesitamos exactamente 39 más para llegar a 87
        try:
            # Puntos de referencia
            ls = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            lk = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            rk = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            la = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            ra = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
            lw = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            rw = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            
            # 1. Inclinación del torso
            torso_inclination = np.arctan2(
                (rs.y + ls.y) - (rh.y + lh.y),
                (rs.x + ls.x) - (rh.x + lh.x)
            )
            features.append(torso_inclination)
            
            # 2. Ángulo de rodilla derecha
            right_knee_angle = np.arctan2(
                ra.y - rk.y, ra.x - rk.x
            ) - np.arctan2(
                rh.y - rk.y, rh.x - rk.x
            )
            features.append(right_knee_angle)
            
            # 3. Ángulo de rodilla izquierda
            left_knee_angle = np.arctan2(
                la.y - lk.y, la.x - lk.x
            ) - np.arctan2(
                lh.y - lk.y, lh.x - lk.x
            )
            features.append(left_knee_angle)
            
            # 4. Ancho de hombros
            shoulder_width = np.sqrt((rs.x - ls.x)**2 + (rs.y - ls.y)**2)
            features.append(shoulder_width)
            
            # 5. Ancho de caderas
            hip_width = np.sqrt((rh.x - lh.x)**2 + (rh.y - lh.y)**2)
            features.append(hip_width)
            
            # 6. Altura del centro de masa
            center_of_mass_height = nose.y - (lh.y + rh.y) / 2
            features.append(center_of_mass_height)
            
            joints_for_velocity = [
                ('LEFT_WRIST', lw), ('RIGHT_WRIST', rw), 
                ('LEFT_ANKLE', la), ('RIGHT_ANKLE', ra),
                ('LEFT_HIP', lh), ('RIGHT_HIP', rh),
                ('LEFT_KNEE', lk), ('RIGHT_KNEE', rk)
            ]
            
            if self.prev_landmarks is not None:
                for joint_name, current_joint in joints_for_velocity:
                    try:
                        prev_joint = self.prev_landmarks[mp.solutions.pose.PoseLandmark[joint_name].value]
                        speed_x = current_joint.x - prev_joint.x
                        speed_y = current_joint.y - prev_joint.y
                        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
                        features.extend([speed_x, speed_y, speed_magnitude, 0.0])  # 4 por articulación
                    except:
                        features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0] * 32)
            
            # Centro de masa y su velocidad
            center_mass_x = (lh.x + rh.x) / 2
            center_mass_y = (lh.y + rh.y) / 2
            if self.prev_landmarks is not None:
                try:
                    prev_lh = self.prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
                    prev_rh = self.prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
                    prev_center_x = (prev_lh.x + prev_rh.x) / 2
                    prev_center_y = (prev_lh.y + prev_rh.y) / 2
                    center_mass_speed = np.sqrt((center_mass_x - prev_center_x)**2 + (center_mass_y - prev_center_y)**2)
                    features.append(center_mass_speed)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
                
        except Exception as e:
            print(f"Error calculando características: {e}")

            while len(features) < 87:
                features.append(0.0)
        
        expected_features = 94
        while len(features) < expected_features:
            features.append(0.0)
        
        if len(features) > expected_features:
            features = features[:expected_features]
        
        print(f"Características generadas: {len(features)}")
        return features
    
    def _calculate_total_movement(self, landmarks):
        """Calcula el movimiento total comparando con el frame anterior"""
        if self.prev_landmarks is None:
            return 0.0
        
        total_movement = 0.0
        key_joints = [
            mp.solutions.pose.PoseLandmark.NOSE,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        for joint in key_joints:
            try:
                current = landmarks[joint.value]
                previous = self.prev_landmarks[joint.value]
                
                distance = np.sqrt(
                    (current.x - previous.x)**2 + 
                    (current.y - previous.y)**2 + 
                    (current.z - previous.z)**2
                )
                total_movement += distance
            except:
                continue
        
        return total_movement / len(key_joints)

def main():
    classifier = ActivityClassifier()
    cap = cv2.VideoCapture(0)
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Iniciando clasificador de actividades...")
    print("Acciones detectables:")
    print("- quieto (cuando hay poco movimiento)")
    print("- caminar_adelante")
    print("- caminar_atras") 
    print("- caminar_derecha")
    print("- caminar_izquierda")
    print("- inclinarse")
    print("- pararse")
    print("- retroceder_frente")
    print("- sentarse")
    print("\nPresiona 'q' para salir")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        activity, confidence, landmarks = classifier.process_frame(frame)
        
        if landmarks:
            # Dibujar landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            if activity and confidence > 0.3:  
                if activity == "quieto":
                    color = (255, 255, 0)
                else:
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                    
                cv2.putText(frame, f"Actividad: {activity}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Confianza: {confidence:.2f}", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "Detectando...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            draw_posture_analysis(frame, landmarks, frame.shape[0], frame.shape[1])
        else:
            cv2.putText(frame, "No se detecta persona", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Reconocimiento de Actividades', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Clasificador cerrado")

if __name__ == "__main__":
    main()