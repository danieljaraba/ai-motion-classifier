from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import argparse

def train_and_evaluate(input_file, model_dir):
    """Entrena y evalúa modelos"""
    df = pd.read_csv(input_file)
    X = df.drop('label', axis=1)
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)
    
    svm = SVC(kernel='rbf', C=10, probability=True)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    print("Reporte de clasificación SVM:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(svm, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
    print(f"Modelos guardados en {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/processed_features.csv', help='Datos procesados')
    parser.add_argument('--model_dir', default='models', help='Directorio para modelos')
    args = parser.parse_args()
    
    train_and_evaluate(args.data, args.model_dir)