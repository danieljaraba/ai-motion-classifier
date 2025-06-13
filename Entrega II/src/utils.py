import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

def calculate_angles(df):
    """Calcula ángulos articulares y características derivadas mejoradas"""
    
    # Inclinación del torso
    df['torso_inclination'] = np.arctan2(
        (df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) - (df['RIGHT_HIP_y'] + df['LEFT_HIP_y']),
        (df['RIGHT_SHOULDER_x'] + df['LEFT_SHOULDER_x']) - (df['RIGHT_HIP_x'] + df['LEFT_HIP_x'])
    )
    
    # Ángulos de rodillas
    df['right_knee_angle'] = np.arctan2(
        df['RIGHT_ANKLE_y'] - df['RIGHT_KNEE_y'],
        df['RIGHT_ANKLE_x'] - df['RIGHT_KNEE_x']
    ) - np.arctan2(
        df['RIGHT_HIP_y'] - df['RIGHT_KNEE_y'],
        df['RIGHT_HIP_x'] - df['RIGHT_KNEE_x']
    )
    
    df['left_knee_angle'] = np.arctan2(
        df['LEFT_ANKLE_y'] - df['LEFT_KNEE_y'],
        df['LEFT_ANKLE_x'] - df['LEFT_KNEE_x']
    ) - np.arctan2(
        df['LEFT_HIP_y'] - df['LEFT_KNEE_y'],
        df['LEFT_HIP_x'] - df['LEFT_KNEE_x']
    )
    
    # Características de postura (sentarse/pararse)
    df['hip_knee_distance'] = np.sqrt(
        (df['RIGHT_HIP_y'] - df['RIGHT_KNEE_y'])**2 + 
        (df['RIGHT_HIP_x'] - df['RIGHT_KNEE_x'])**2
    )
    
    df['knee_ankle_distance'] = np.sqrt(
        (df['RIGHT_KNEE_y'] - df['RIGHT_ANKLE_y'])**2 + 
        (df['RIGHT_KNEE_x'] - df['RIGHT_ANKLE_x'])**2
    )
    
    # Altura relativa del centro de masa
    df['center_of_mass_height'] = df['NOSE_y'] - (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2
    
    # Ancho de hombros y caderas
    df['shoulder_width'] = np.sqrt(
        (df['RIGHT_SHOULDER_x'] - df['LEFT_SHOULDER_x'])**2 + 
        (df['RIGHT_SHOULDER_y'] - df['LEFT_SHOULDER_y'])**2
    )
    
    df['hip_width'] = np.sqrt(
        (df['RIGHT_HIP_x'] - df['LEFT_HIP_x'])**2 + 
        (df['RIGHT_HIP_y'] - df['LEFT_HIP_y'])**2
    )
    
    # Características para detectar dirección de movimiento
    df['body_orientation'] = np.arctan2(
        df['NOSE_y'] - (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2,
        df['NOSE_x'] - (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
    )
    
    # Diferencia de altura entre pies (para detectar inclinación)
    df['feet_height_diff'] = df['LEFT_ANKLE_y'] - df['RIGHT_ANKLE_y']
    
    # Posición relativa de manos
    df['hands_relative_height'] = (df['LEFT_WRIST_y'] + df['RIGHT_WRIST_y']) / 2 - (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2
    
    return df

def calculate_velocity_features(df):
    """Calcula velocidades para coincidir exactamente con realtime_classifier (87 características total)"""
    
    # Velocidades para 8 articulaciones específicas (32 características)
    joints_for_velocity = ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ANKLE', 'RIGHT_ANKLE', 
                          'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
    
    for joint in joints_for_velocity:
        df[f'{joint}_speed_x'] = df.groupby('label')[f'{joint}_x'].diff().fillna(0)
        df[f'{joint}_speed_y'] = df.groupby('label')[f'{joint}_y'].diff().fillna(0)
        df[f'{joint}_speed'] = np.sqrt(
            df[f'{joint}_speed_x']**2 + df[f'{joint}_speed_y']**2
        )
        # Cuarta característica (placeholder)
        df[f'{joint}_speed_placeholder'] = 0.0
    
    # Velocidad del centro de masa (1 característica)
    df['center_mass_x'] = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
    df['center_mass_y'] = (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2
    df['center_mass_speed'] = np.sqrt(
        df.groupby('label')['center_mass_x'].diff().fillna(0)**2 + 
        df.groupby('label')['center_mass_y'].diff().fillna(0)**2
    )
    
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
        'RIGHT_HIP': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        'LEFT_KNEE': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        'RIGHT_KNEE': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        'LEFT_ANKLE': mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        'RIGHT_ANKLE': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
        'NOSE': mp.solutions.pose.PoseLandmark.NOSE
    }
    
    # Líneas de referencia
    for pair in [('LEFT_SHOULDER', 'RIGHT_SHOULDER'), 
                 ('LEFT_HIP', 'RIGHT_HIP'),
                 ('LEFT_KNEE', 'RIGHT_KNEE')]:
        p1 = landmarks.landmark[joints[pair[0]].value]
        p2 = landmarks.landmark[joints[pair[1]].value]
        cv2.line(frame, norm_to_pixel(p1.x, p1.y), norm_to_pixel(p2.x, p2.y), 
                (0, 255, 255), 2)
    
    # Análisis de postura
    ls = landmarks.landmark[joints['LEFT_SHOULDER'].value]
    rs = landmarks.landmark[joints['RIGHT_SHOULDER'].value]
    lh = landmarks.landmark[joints['LEFT_HIP'].value]
    rh = landmarks.landmark[joints['RIGHT_HIP'].value]
    lk = landmarks.landmark[joints['LEFT_KNEE'].value]
    rk = landmarks.landmark[joints['RIGHT_KNEE'].value]
    la = landmarks.landmark[joints['LEFT_ANKLE'].value]
    ra = landmarks.landmark[joints['RIGHT_ANKLE'].value]
    nose = landmarks.landmark[joints['NOSE'].value]
    
    # Centro de masa y orientación
    shoulder_center = ((ls.x + rs.x)/2, (ls.y + rs.y)/2)
    hip_center = ((lh.x + rh.x)/2, (lh.y + rh.y)/2)
    
    # Inclinación del torso
    torso_angle = np.degrees(np.arctan2(
        shoulder_center[1] - hip_center[1],
        shoulder_center[0] - hip_center[0]
    ))
    
    # Altura relativa (para detectar sentarse/pararse)
    relative_height = nose.y - hip_center[1]
    
    # Diferencia de altura entre pies
    feet_diff = abs(la.y - ra.y)
    
    # Mostrar información
    cv2.putText(frame, f"Torso: {torso_angle:.1f}°", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Altura rel: {relative_height:.3f}", (20, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Pies diff: {feet_diff:.3f}", (20, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Línea del torso
    cv2.line(frame, norm_to_pixel(shoulder_center[0], shoulder_center[1]), 
             norm_to_pixel(hip_center[0], hip_center[1]), (255, 0, 0), 3)