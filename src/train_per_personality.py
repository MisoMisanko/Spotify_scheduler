# src/train_per_personality.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_FILE = os.path.join("data", "figshare_PER", "PER_dataset.xls")
MODEL_FILE = os.path.join("models", "per_personality.pkl")

# Big Five traits in PER dataset
BIG5 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Candidate audio feature columns in PER
AUDIO_FEATURES = [
    "rms", "lowenergy", "eventdensity", "pulseclarity",
    "attacktime_mean", "attacktime_std", "attacktime_slope", "attacktime_periodentropy",
    "attackslope_mean", "attackslope_std", "attackslope_slope", "attackslope_periodentropy",
    "zerocross", "brightness", "centroid", "spread", "skewness", "kurtosis",
    "flatness", "mirentropy", "inharmonicity", "activity", "valence", "tension",
    "happy", "sad", "tender", "anger", "fear"
]

def explore_data(df, features, traits):
    """Explore data quality and feature-target relationships"""
    print("\n🔍 DATA EXPLORATION")
    print(f"Dataset shape: {df.shape}")
    print(f"Available features: {len(features)}")
    
    # Check for missing values
    missing = df[features + traits].isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check feature variance
    feature_vars = df[features].var()
    low_var_features = feature_vars[feature_vars < 0.001].index.tolist()
    if low_var_features:
        print(f"Low variance features: {low_var_features}")
    
    # Basic stats for personality traits
    print("\nPersonality trait statistics:")
    for trait in traits:
        if trait in df.columns:
            stats = df[trait].describe()
            print(f"{trait}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")

def preprocess_features(X, y, feature_selection=True, n_features=15):
    """Preprocess features with scaling and optional selection"""
    # Remove low variance features
    feature_vars = X.var()
    high_var_features = feature_vars[feature_vars >= 0.001].index
    X_filtered = X[high_var_features]
    
    print(f"Removed {len(X.columns) - len(high_var_features)} low-variance features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filtered), 
        columns=X_filtered.columns, 
        index=X_filtered.index
    )
    
    # Feature selection
    if feature_selection and len(X_scaled.columns) > n_features:
        selector = SelectKBest(score_func=f_regression, k=min(n_features, len(X_scaled.columns)))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X_scaled.columns[selector.get_support()].tolist()
        X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
        print(f"Selected top {len(selected_features)} features: {selected_features[:5]}...")
        return X_final, scaler, selector, selected_features
    
    return X_scaled, scaler, None, X_scaled.columns.tolist()

def train_model_with_validation(X, y, trait_name):
    """Train model with proper validation"""
    # Split for final validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Try different hyperparameters
    param_configs = [
        {"n_estimators": 100, "max_depth": 3, "min_samples_split": 20, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 3},
    ]
    
    best_score = -np.inf
    best_model = None
    best_config = None
    
    for config in param_configs:
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **config)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        avg_score = cv_scores.mean()
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_config = config
    
    # Train best model on full training set
    best_model.fit(X_train, y_train)
    
    # Final validation
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate baseline (predicting mean)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    print(f"{trait_name}:")
    print(f"  Best config: {best_config}")
    print(f"  CV R²: {best_score:.3f}")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}")
    print(f"  Baseline R²: {baseline_r2:.3f}")
    print(f"  Improvement: {test_r2 - baseline_r2:.3f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(5)
        print(f"  Top features: {dict(top_features.round(3))}")
    
    return best_model, test_r2

def main():
    print(f"📂 Loading {DATA_FILE} ...")
    df = pd.read_excel(DATA_FILE)

    # Filter available features and traits
    available_features = [c for c in AUDIO_FEATURES if c in df.columns]
    available_traits = [c for c in BIG5 if c in df.columns]
    
    if not available_features:
        raise ValueError("No audio features found in PER dataset!")
    if not available_traits:
        raise ValueError("No personality traits found in PER dataset!")

    print(f"🎵 Found {len(available_features)} audio features")
    print(f"🧠 Found {len(available_traits)} personality traits: {available_traits}")

    # Explore data
    explore_data(df, available_features, available_traits)

    # Remove rows with missing personality scores
    df_clean = df.dropna(subset=available_traits)
    print(f"\nCleaned dataset: {df_clean.shape[0]} samples")

    X_raw = df_clean[available_features]
    models = {}
    scalers = {}
    selectors = {}
    feature_lists = {}

    print("\n🤖 TRAINING MODELS")
    print("="*50)

    for trait in available_traits:
        y = df_clean[trait]
        
        # Preprocess features
        X_processed, scaler, selector, selected_features = preprocess_features(
            X_raw, y, feature_selection=True, n_features=10
        )
        
        # Train model
        model, test_r2 = train_model_with_validation(X_processed, y, trait)
        
        # Store everything
        models[trait] = model
        scalers[trait] = scaler
        selectors[trait] = selector
        feature_lists[trait] = selected_features
        
        print("-" * 30)

    # Save everything
    os.makedirs("models", exist_ok=True)
    model_data = {
        "models": models,
        "scalers": scalers,
        "selectors": selectors,
        "feature_lists": feature_lists,
        "all_features": available_features
    }
    
    joblib.dump(model_data, MODEL_FILE)
    print(f"\n✅ Saved comprehensive model → {MODEL_FILE}")

if __name__ == "__main__":
    main()