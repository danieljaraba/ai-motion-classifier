import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

def calculate_angles(df):
    """Calcula ángulos articulares y características derivadas"""
    df['torso_inclination'] = np.arctan2(
        (df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) - (df['RIGHT_HIP_y'] + df['LEFT_HIP_y']),
        (df['RIGHT_SHOULDER_x'] + df['LEFT_SHOULDER_x']) - (df['RIGHT_HIP_x'] + df['LEFT_HIP_x'])
    )
    
    df['right_knee_angle'] = np.arctan2(
        df['RIGHT_ANKLE_y'] - df['RIGHT_KNEE_y'],
        df['RIGHT_ANKLE_x'] - df['RIGHT_KNEE_x']
    ) - np.arctan2(
        df['RIGHT_HIP_y'] - df['RIGHT_KNEE_y'],
        df['RIGHT_HIP_x'] - df['RIGHT_KNEE_x']
    )
    
    return df

def calculate_velocity_features(df):
    """Calcula velocidades de movimiento entre frames"""
    for joint in ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ANKLE', 'RIGHT_ANKLE']:
        df[f'{joint}_speed'] = np.sqrt(
            df.groupby('label')[f'{joint}_x'].diff()**2 + 
            df.groupby('label')[f'{joint}_y'].diff()**2
        ).fillna(0)
    
    return df

def draw_posture_analysis(frame, landmarks, height=480, width=640):
    """Dibuja análisis postural en el frame"""

    def norm_to_pixel(x, y):
        return int(x * width), int(y * height)
    
    # Puntos clave
    joints = {
        'LEFT_SHOULDER': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        'RIGHT_SHOULDER': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        'LEFT_HIP': mp.solutions.pose.PoseLandmark.LEFT_HIP,
        'RIGHT_HIP': mp.solutions.pose.PoseLandmark.RIGHT_HIP
    }
    
    for pair in [('LEFT_SHOULDER', 'RIGHT_SHOULDER'), 
                 ('LEFT_HIP', 'RIGHT_HIP')]:
        p1 = landmarks.landmark[joints[pair[0]].value]
        p2 = landmarks.landmark[joints[pair[1]].value]
        cv2.line(frame, norm_to_pixel(p1.x, p1.y), norm_to_pixel(p2.x, p2.y), 
                (0, 255, 255), 2)
    
    ls = landmarks.landmark[joints['LEFT_SHOULDER'].value]
    rs = landmarks.landmark[joints['RIGHT_SHOULDER'].value]
    lh = landmarks.landmark[joints['LEFT_HIP'].value]
    rh = landmarks.landmark[joints['RIGHT_HIP'].value]
    
    shoulder_center = ((ls.x + rs.x)/2, (ls.y + rs.y)/2)
    hip_center = ((lh.x + rh.x)/2, (lh.y + rh.y)/2)
    
    angle = np.degrees(np.arctan2(
        shoulder_center[1] - hip_center[1],
        shoulder_center[0] - hip_center[0]
    ))
    
    cv2.putText(frame, f"Inclinacion: {angle:.1f}°", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)