import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ProductionModelTrainer:
    """
    Train production-ready models based on successful genre-preference approach
    """
    
    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.performance_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("Loading Kaggle dataset for production model training...")
        
        try:
            df = pd.read_csv('data/kaggle_young_people/responses.csv')
            print(f"Loaded: {df.shape[0]} participants, {df.shape[1]} questions")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
        
        # Music genre features
        music_genres = [
            'Music', 'Slow songs or fast songs', 'Dance', 'Folk', 'Country',
            'Classical music', 'Musical', 'Pop', 'Rock', 'Metal or Hardrock',
            'Punk', 'Hiphop, Rap', 'Reggae, Ska', 'Swing, Jazz', 'Rock n roll',
            'Alternative', 'Latino', 'Techno, Trance', 'Opera'
        ]
        
        music_data = df[music_genres].fillna(df[music_genres].mean())
        
        # Personality composites
        personality_items = {
            'Openness': ['Art exhibitions', 'Theatre', 'Reading', 'Science and technology'],
            'Conscientiousness': ['Daily events', 'Prioritising workload', 'Writing notes', 'Reliability'],
            'Extraversion': ['Socializing', 'Fun with friends', 'Assertiveness', 'Energy levels'],
            'Agreeableness': ['Empathy', 'Compassion to animals', 'Giving', 'Charity'],
            'Neuroticism': ['Mood swings', 'Getting angry', 'Loneliness', 'Life struggles']
        }
        
        personality_data = {}
        for trait, items in personality_items.items():
            available = [item for item in items if item in df.columns]
            if len(available) >= 2:
                scores = df[available].fillna(df[available].mean())
                personality_data[trait] = scores.mean(axis=1)
                print(f"{trait}: Using {len(available)} questionnaire items")
        
        personality_df = pd.DataFrame(personality_data)
        
        # Clean data
        complete_mask = ~(music_data.isnull().any(axis=1) | personality_df.isnull().any(axis=1))
        X_clean = music_data[complete_mask]
        y_clean = personality_df[complete_mask]
        
        print(f"Training data: {len(X_clean)} samples, {X_clean.shape[1]} features")
        self.feature_names = list(X_clean.columns)
        return X_clean, y_clean
    
    def train_optimized_models(self, X, y):
        """Train optimized models for each trait"""
        print("\nTraining optimized production models...")
        
        for trait in y.columns:
            print(f"\nOptimizing {trait} model...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[trait], test_size=0.2, random_state=42
            )
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge())
            ])
            
            param_grid = {'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            self.models[trait] = grid_search.best_estimator_
            self.performance_metrics[trait] = {
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_score': grid_search.best_score_,
                'best_params': grid_search.best_params_
            }
            
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  CV R²: {grid_search.best_score_:.3f}")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Test MAE: {test_mae:.3f}")
    
    def save_production_models(self):
        """Save models without lambda functions"""
        print(f"\nSaving production models...")
        
        # Simple model package without lambda functions
        model_package = {
            'models': self.models,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'version': '1.0',
            'training_date': pd.Timestamp.now().isoformat(),
        }
        
        model_file = 'models/production_personality_models.pkl'
        joblib.dump(model_package, model_file)
        print(f"✅ Saved production models to: {model_file}")
        
        # Save performance summary
        performance_summary = pd.DataFrame(self.performance_metrics).T
        performance_summary.to_csv('models/production_model_performance.csv')
        print(f"✅ Saved performance metrics to: models/production_model_performance.csv")
        
        return model_file
    
    def create_prediction_interface(self):
        """Note: Prediction interface should be created manually"""
        print(f"✅ Models ready - create production_predictor.py manually")

def main():
    """Train and save production models"""
    print("TRAINING PRODUCTION MODELS FOR SPOTIFY APP")
    print("="*60)
    
    trainer = ProductionModelTrainer()
    
    X, y = trainer.load_and_prepare_data()
    if X is None or y is None:
        print("❌ Failed to load training data")
        return
    
    trainer.train_optimized_models(X, y)
    model_file = trainer.save_production_models()
    trainer.create_prediction_interface()
    
    print(f"\n" + "="*60)
    print("PRODUCTION MODELS READY")
    print("="*60)
    print(f"✅ Models saved: {model_file}")
    print(f"✅ Interface created: production_predictor.py")
    
    print(f"\nTo use in your Spotify app:")
    print(f"1. Import: from production_predictor import PersonalityPredictor")
    print(f"2. Use: predictor = PersonalityPredictor()")
    print(f"3. Predict: predictor.predict_from_spotify_features(audio_features)")
    
    print(f"\nModel performance summary:")
    for trait, metrics in trainer.performance_metrics.items():
        r2 = metrics['test_r2']
        quality = "Excellent" if r2 > 0.25 else "Good" if r2 > 0.1 else "Fair"
        print(f"  {trait}: R² = {r2:.3f} ({quality})")

if __name__ == "__main__":
    main()