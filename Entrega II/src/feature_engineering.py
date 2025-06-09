import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
from utils import calculate_velocity_features

def preprocess_data(input_file, output_dir):
    """Preprocesa los datos para entrenamiento"""
    df = pd.read_csv(input_file).fillna(0)
    
    df = calculate_velocity_features(df)
    
    X = df.drop(['label', 'frame_num'], axis=1)
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(output_dir, 'pca.pkl'))
    
    processed_df = pd.DataFrame(X_pca)
    processed_df['label'] = y.values
    processed_df.to_csv(os.path.join(output_dir, 'processed_features.csv'), index=False)
    
    print(f"Procesamiento completado. Datos guardados en {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset/raw_landmarks.csv', help='Archivo CSV con landmarks')
    parser.add_argument('--output_dir', default='dataset', help='Directorio para salida')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output_dir)