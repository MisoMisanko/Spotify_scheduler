# production_predictor.py - Clean personality predictor for Spotify app

import joblib
import pandas as pd
import numpy as np

class PersonalityPredictor:
    """Production personality predictor for Spotify app"""
    
    def __init__(self, model_file='models/production_personality_models.pkl'):
        try:
            self.model_data = joblib.load(model_file)
            self.models = self.model_data['models']
            self.feature_names = self.model_data['feature_names']
            print(f"Loaded models for: {list(self.models.keys())}")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = None
    
    def predict_from_genres(self, genre_preferences):
        """
        Predict personality from genre preferences (1-5 scale)
        genre_preferences: dict with genre names and ratings
        """
        if not self.models:
            return {'error': 'Models not loaded'}
        
        # Create feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            value = genre_preferences.get(feature_name, 3.0)  # Default neutral
            feature_vector.append(value)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Predict each trait
        predictions = {}
        for trait, model in self.models.items():
            try:
                pred = model.predict(feature_array)[0]
                predictions[trait] = round(np.clip(pred, 1.0, 5.0), 2)
            except Exception as e:
                predictions[trait] = 3.0  # Default if prediction fails
        
        return predictions
    
    def predict_from_spotify_features(self, spotify_features):
        """
        Predict personality from Spotify audio features
        Maps audio features to estimated genre preferences
        """
        # Extract key Spotify features with defaults
        danceability = spotify_features.get('danceability', 0.5)
        energy = spotify_features.get('energy', 0.5)
        valence = spotify_features.get('valence', 0.5)
        acousticness = spotify_features.get('acousticness', 0.5)
        instrumentalness = spotify_features.get('instrumentalness', 0.1)
        loudness = spotify_features.get('loudness', -10)
        speechiness = spotify_features.get('speechiness', 0.1)
        
        # Normalize loudness to 0-1 scale
        loudness_norm = max(0, min(1, (loudness + 60) / 60))
        
        # Map to genre preferences using audio feature patterns
        genre_estimates = {
            # Electronic/Dance music
            'Dance': min(5, max(1, danceability * 4 + energy * 1)),
            'Techno, Trance': min(5, max(1, danceability * 3 + energy * 2)),
            
            # Classical/Orchestral
            'Classical music': min(5, max(1, acousticness * 3 + instrumentalness * 2)),
            'Opera': min(5, max(1, acousticness * 2.5 + instrumentalness * 1.5 + (1-danceability) * 1)),
            
            # Rock/Metal
            'Rock': min(5, max(1, energy * 2 + loudness_norm * 2 + (1-acousticness) * 1)),
            'Metal or Hardrock': min(5, max(1, energy * 2.5 + loudness_norm * 2.5)),
            
            # Pop/Mainstream
            'Pop': min(5, max(1, valence * 2 + danceability * 1.5 + energy * 1.5)),
            'Musical': min(5, max(1, valence * 2 + (1-instrumentalness) * 2 + energy * 1)),
            
            # Folk/Acoustic
            'Folk': min(5, max(1, acousticness * 3 + (1-danceability) * 1.5 + (1-energy) * 0.5)),
            'Country': min(5, max(1, acousticness * 2 + (1-danceability) * 2 + valence * 1)),
            
            # Jazz/Blues
            'Swing, Jazz': min(5, max(1, instrumentalness * 2 + (1-speechiness) * 2 + energy * 1)),
            
            # Hip-hop/Rap
            'Hiphop, Rap': min(5, max(1, speechiness * 4 + energy * 1)),
            
            # Other genres - set to neutral
            'Music': 3.0,
            'Slow songs or fast songs': 5 if energy < 0.4 else 1,  # Slow if low energy
            'Punk': min(5, max(1, energy * 2.5 + loudness_norm * 1.5)),
            'Reggae, Ska': min(5, max(1, danceability * 2 + valence * 2)),
            'Rock n roll': min(5, max(1, energy * 2 + valence * 1.5 + loudness_norm * 0.5)),
            'Alternative': min(5, max(1, energy * 1.5 + (1-valence) * 1 + loudness_norm * 1.5)),
            'Latino': min(5, max(1, danceability * 2.5 + valence * 1.5))
        }
        
        return self.predict_from_genres(genre_estimates)
    
    def get_model_info(self):
        """Get model performance information"""
        if not self.models:
            return {'error': 'Models not loaded'}
        
        performance = self.model_data['performance_metrics']
        
        info = {
            'version': self.model_data['version'],
            'training_date': self.model_data['training_date'],
            'traits': list(self.models.keys()),
            'performance': {}
        }
        
        for trait in performance:
            r2 = performance[trait]['test_r2']
            info['performance'][trait] = {
                'r2_score': round(r2, 3),
                'accuracy': 'Excellent' if r2 > 0.25 else 'Good' if r2 > 0.1 else 'Fair'
            }
        
        return info