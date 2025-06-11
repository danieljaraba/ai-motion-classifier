import joblib
import pandas as pd
import numpy as np

def diagnose_model(model_dir='models'):
    """Diagnose the trained model and scaler to understand expected features"""
    
    print("=== MODEL DIAGNOSTIC ===")
    
    # Load the scaler
    try:
        scaler = joblib.load(f'{model_dir}/scaler.pkl')
        print(f"Scaler expects {scaler.n_features_in_} features")
        print(f"Scaler feature names: {getattr(scaler, 'feature_names_in_', 'Not available')}")
        
        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            print("\nExpected feature names:")
            for i, name in enumerate(scaler.feature_names_in_):
                print(f"  {i}: {name}")
        
    except Exception as e:
        print(f"Error loading scaler: {e}")
    
    # Load the PCA
    try:
        pca = joblib.load(f'{model_dir}/pca.pkl')
        print(f"\nPCA expects {pca.n_features_in_} features")
        print(f"PCA reduces to {pca.n_components_} components")
    except Exception as e:
        print(f"Error loading PCA: {e}")
    
    # Load the model
    try:
        model = joblib.load(f'{model_dir}/svm_model.pkl')
        print(f"\nModel type: {type(model)}")
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Load label encoder
    try:
        le = joblib.load(f'{model_dir}/label_encoder.pkl')
        print(f"\nLabel encoder classes: {le.classes_}")
    except Exception as e:
        print(f"Error loading label encoder: {e}")

def analyze_training_data(data_file='dataset/processed_features.csv'):
    """Analyze the training data to understand feature structure"""
    
    print("\n=== TRAINING DATA ANALYSIS ===")
    
    try:
        df = pd.read_csv(data_file)
        print(f"Training data shape: {df.shape}")
        print(f"Number of feature columns: {len([col for col in df.columns if col not in ['label', 'frame_num']])}")
        
        feature_cols = [col for col in df.columns if col not in ['label', 'frame_num']]
        print("\nFeature columns:")
        for i, col in enumerate(feature_cols):
            print(f"  {i}: {col}")
            
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        
    except Exception as e:
        print(f"Error loading training data: {e}")

def create_feature_mapping():
    """Create a mapping of expected features based on the training pipeline"""
    
    print("\n=== EXPECTED FEATURE STRUCTURE ===")
    
    # Base joint features (12 joints × 4 features each = 48 features)
    joints = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_WRIST', 'RIGHT_WRIST', 'NOSE', 'LEFT_FOOT_INDEX'
    ]
    
    features = []
    
    # Coordinate features
    for joint in joints:
        features.extend([
            f'{joint}_x',
            f'{joint}_y', 
            f'{joint}_z',
            f'{joint}_visibility'
        ])
    
    print(f"Base coordinate features: {len(features)}")
    
    # Derived features (based on utils.py)
    derived_features = [
        'torso_inclination',
        'right_knee_angle',
        'left_knee_angle',  # Likely added
        'shoulder_width',   # Likely added
        'hip_width',        # Likely added
        'center_of_mass'    # Likely added
    ]
    
    features.extend(derived_features)
    
    print(f"Total expected features: {len(features)}")
    print("\nComplete feature list:")
    for i, feature in enumerate(features):
        print(f"  {i}: {feature}")
    
    return features

if __name__ == "__main__":
    diagnose_model()
    analyze_training_data()
    create_feature_mapping()