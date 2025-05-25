import re
import pickle
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class PhoneRecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.models = {}
        self.label_encoders = {}
        self.feature_names = ['os', 'price', 'ram', 'battery', 'storage', 'camera', 'brand', 'usage', 'screen']
        
    def preprocess_text(self, text):
        # Türkçe karakterleri normalize et
        text = text.lower()
        text = text.replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
        text = text.replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
        return text
    
    def extract_features_from_output(self, output_text):
        features = {}
        
        # Output formatını parse et
        parts = output_text.split(';')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                features[key] = value
        
        return features
    
    def prepare_training_data(self, data_file):
        X = []  # Input texts
        y_dict = {feature: [] for feature in self.feature_names}
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if ' -> ' in line:
                input_text, output_text = line.split(' -> ', 1)
                
                # Input text preprocessing
                processed_input = self.preprocess_text(input_text)
                X.append(processed_input)
                
                # Extract features from output
                features = self.extract_features_from_output(output_text)
                
                # Fill feature vectors
                for feature in self.feature_names:
                    value = features.get(feature, 'none')
                    y_dict[feature].append(value)
        
        return X, y_dict
    
    def train(self, data_file):
        print("Eğitim verisi yükleniyor...")
        X, y_dict = self.prepare_training_data(data_file)
        
        print(f"Toplam {len(X)} örnek yüklendi.")
        
        # Text vectorization
        print("Metin vektörleştiriliyor...")
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train separate models for each feature
        for feature in self.feature_names:
            print(f"{feature} özelliği için model eğitiliyor...")
            
            # Label encoding
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_dict[feature])
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_vectorized, y_encoded)
            
            # Store model and encoder
            self.models[feature] = model
            self.label_encoders[feature] = le
        
        print("Model eğitimi tamamlandı!")
    
    def save_model(self, model_path):
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model {model_path} konumuna kaydedildi.")

if __name__ == "__main__":
    # Model oluştur ve eğit
    model = PhoneRecommendationModel()
    model.train('training_data.txt')
    model.save_model('phone_model.pkl')
    
    print("Eğitim tamamlandı! Model 'phone_model.pkl' dosyasına kaydedildi.")
