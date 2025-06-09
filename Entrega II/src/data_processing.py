import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import os
from utils import calculate_angles 

mp_pose = mp.solutions.pose

def extract_landmarks(video_path, label):
    """Extrae landmarks de un video y los devuelve como DataFrame"""
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = {'label': label, 'frame_num': int(cap.get(cv2.CAP_PROP_POS_FRAMES))}
            
            joints = [
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
            
            for joint in joints:
                joint_point = landmarks[mp_pose.PoseLandmark[joint].value]
                
                frame_data.update({
                    f'{joint}_x': joint_point.x,
                    f'{joint}_y': joint_point.y,
                    f'{joint}_z': joint_point.z
                })

                frame_data[f'{joint}_visibility'] = joint_point.visibility
            
            data.append(frame_data)
    
    cap.release()
    return pd.DataFrame(data)

def process_all_videos(input_dir, output_file):
    """Procesa todos los videos en un directorio"""
    all_data = []
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.mp4'):
            label = filename.split('_')[0]
            video_path = os.path.join(input_dir, filename)
            df = extract_landmarks(video_path, label)
            df = calculate_angles(df)
            all_data.append(df)
    
    full_dataset = pd.concat(all_data)
    full_dataset.to_csv(output_file, index=False)
    print(f"Datos guardados en {output_file}")
    print(f"Número de columnas de features: {len([col for col in full_dataset.columns if col not in ['label', 'frame_num']])}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data', help='Directorio con videos de entrada')
    parser.add_argument('--output', default='dataset/raw_landmarks.csv', help='Archivo CSV de salida')
    args = parser.parse_args()
    
    os.makedirs('dataset', exist_ok=True)
    process_all_videos(args.input_dir, args.output)