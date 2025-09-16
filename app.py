# app.py - DEBUG VERSION to identify why all users get same results
import os
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import requests
import json
import joblib
from sklearn.preprocessing import StandardScaler

# Spotify credentials
SCOPES = (
    "user-top-read "
    "playlist-read-private "
    "user-library-read "
    "user-follow-read "
    "user-read-recently-played"
)

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

# Last.fm credentials
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_SHARED_SECRET = os.environ.get("LASTFM_SHARED_SECRET")

st.set_page_config(
    page_title="Spotify Personality Predictor - DEBUG",
    page_icon="🐛",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

# DEBUG: Add session state to track user-specific data
if 'debug_data' not in st.session_state:
    st.session_state.debug_data = {}

# Authentication functions (same as before)
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        open_browser=False,
        cache_path=None,
    )

def ensure_spotify_client():
    auth_manager = get_auth_manager()
    token_info = st.session_state.get("token_info")

    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    params = st.query_params
    if "code" in params:
        code = params["code"]
        token_info = auth_manager.get_access_token(code, as_dict=True)
        st.session_state["token_info"] = token_info
        st.query_params.clear()
        st.rerun()

    login_url = auth_manager.get_authorize_url()
    st.info("Please log in with Spotify")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

@st.cache_data
def load_trained_models():
    """Load pre-trained personality prediction models"""
    
    model_files = {
        'behavioral': 'models/behavioral_personality_models.pkl',
        'research': 'models/research_personality_models.pkl',
        'per': 'models/per_personality.pkl'
    }
    
    loaded_models = {}
    
    for model_name, file_path in model_files.items():
        try:
            if os.path.exists(file_path):
                model_data = joblib.load(file_path)
                loaded_models[model_name] = model_data
                st.sidebar.success(f"✅ Loaded {model_name} model")
            else:
                st.sidebar.warning(f"⚠️ {model_name} model not found at {file_path}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {model_name} model: {e}")
    
    return loaded_models

def get_user_music_data_debug(sp, limit=50):
    """DEBUG: Get user's music data with detailed logging"""
    
    st.write("🐛 DEBUG: Starting music data collection...")
    
    # Get user info to differentiate users
    try:
        user_info = sp.current_user()
        user_id = user_info.get('id', 'unknown_user')
        st.write(f"🐛 DEBUG: User ID: {user_id}")
        st.session_state.debug_data['user_id'] = user_id
    except Exception as e:
        st.write(f"🐛 DEBUG: Could not get user info: {e}")
        user_id = "unknown_user"
    
    music_data = {
        'user_id': user_id,
        'tracks': [],
        'artists': [],
        'genres': [],
        'track_names': [],
        'artist_names': []
    }
    
    try:
        st.write("🐛 DEBUG: Fetching tracks from different time periods...")
        
        all_tracks = []
        
        # Get tracks from multiple sources with detailed logging
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                tracks = sp.current_user_top_tracks(limit=limit, time_range=time_range)
                track_count = len(tracks['items'])
                all_tracks.extend(tracks['items'])
                
                # DEBUG: Show track names for verification
                track_names = [track['name'] for track in tracks['items'][:3]]
                st.write(f"🐛 DEBUG: {time_range} - {track_count} tracks. Sample: {track_names}")
                
            except Exception as e:
                st.write(f"🐛 DEBUG: Error getting {time_range} tracks: {e}")
        
        # Get recent tracks
        try:
            recent = sp.current_user_recently_played(limit=limit)
            recent_tracks = [item['track'] for item in recent['items']]
            recent_count = len(recent_tracks)
            all_tracks.extend(recent_tracks)
            
            # DEBUG: Show recent track names
            recent_names = [track['name'] for track in recent_tracks[:3]]
            st.write(f"🐛 DEBUG: Recent tracks - {recent_count} tracks. Sample: {recent_names}")
            
        except Exception as e:
            st.write(f"🐛 DEBUG: Error getting recent tracks: {e}")
        
        if not all_tracks:
            st.error("🐛 DEBUG: No tracks found!")
            return None
        
        # Remove duplicates and collect names for verification
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track['id'] and track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        music_data['tracks'] = unique_tracks
        music_data['track_names'] = [track['name'] for track in unique_tracks]
        music_data['artist_names'] = [track['artists'][0]['name'] for track in unique_tracks]
        
        st.write(f"🐛 DEBUG: Found {len(unique_tracks)} unique tracks")
        st.write(f"🐛 DEBUG: Sample track names: {music_data['track_names'][:5]}")
        st.write(f"🐛 DEBUG: Sample artists: {music_data['artist_names'][:5]}")
        
        # Get artist info for genres
        artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks if track['artists']]))
        st.write(f"🐛 DEBUG: Getting info for {len(artist_ids)} unique artists...")
        
        all_artists = []
        all_genres = []
        
        for i in range(0, len(artist_ids), 20):
            batch_ids = artist_ids[i:i+20]
            try:
                artists_response = sp.artists(batch_ids)
                batch_artists = artists_response['artists']
                all_artists.extend(batch_artists)
                
                for artist in batch_artists:
                    artist_genres = artist.get('genres', [])
                    all_genres.extend(artist_genres)
                
                time.sleep(0.1)
                
            except Exception as e:
                st.write(f"🐛 DEBUG: Error getting artist batch: {e}")
        
        music_data['artists'] = all_artists
        music_data['genres'] = all_genres
        
        st.write(f"🐛 DEBUG: Found {len(all_genres)} genre tags")
        st.write(f"🐛 DEBUG: Sample genres: {list(set(all_genres))[:10]}")
        
        # Store in session state for comparison
        st.session_state.debug_data['music_data'] = music_data
        
        return music_data
        
    except Exception as e:
        st.error(f"🐛 DEBUG: Error in music data collection: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_features_from_metadata_debug(music_data):
    """DEBUG: Create features with detailed logging"""
    
    st.write("🐛 DEBUG: Starting feature extraction...")
    
    tracks = music_data['tracks']
    genres = music_data['genres']
    user_id = music_data.get('user_id', 'unknown')
    
    if not tracks:
        st.error("🐛 DEBUG: No tracks to analyze!")
        return None
    
    st.write(f"🐛 DEBUG: Processing {len(tracks)} tracks for user {user_id}")
    
    # Analyze track metadata with detailed logging
    track_data = []
    for track in tracks:
        track_info = {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'popularity': track.get('popularity', 50),
            'explicit': track.get('explicit', False),
            'duration_ms': track.get('duration_ms', 180000)
        }
        track_data.append(track_info)
    
    track_df = pd.DataFrame(track_data)
    
    # DEBUG: Show actual values being calculated
    st.write("🐛 DEBUG: Track statistics:")
    st.write(f"  - Average popularity: {track_df['popularity'].mean():.2f}")
    st.write(f"  - Explicit tracks: {track_df['explicit'].sum()}/{len(tracks)}")
    st.write(f"  - Average duration: {track_df['duration_ms'].mean()/60000:.2f} minutes")
    
    # Get enriched genre/tag data
    all_genres = genres
    genre_text = ' '.join(all_genres).lower()
    
    st.write(f"🐛 DEBUG: Genre analysis:")
    st.write(f"  - Total genre mentions: {len(all_genres)}")
    st.write(f"  - Unique genres: {len(set(all_genres))}")
    st.write(f"  - Sample genres: {list(set(all_genres))[:5]}")
    
    # Create features with detailed calculation logging
    features = {}
    
    # Basic popularity and mainstream metrics
    popularity_mean = track_df['popularity'].mean() / 100
    features['popularity'] = popularity_mean
    features['mainstream_preference'] = (track_df['popularity'] > 70).mean()
    features['underground_preference'] = (track_df['popularity'] < 30).mean()
    features['explicit_content_ratio'] = track_df['explicit'].mean()
    
    st.write(f"🐛 DEBUG: Basic metrics calculated:")
    st.write(f"  - Popularity: {features['popularity']:.3f}")
    st.write(f"  - Mainstream: {features['mainstream_preference']:.3f}")
    st.write(f"  - Underground: {features['underground_preference']:.3f}")
    st.write(f"  - Explicit ratio: {features['explicit_content_ratio']:.3f}")
    
    # Duration preferences
    avg_duration_min = track_df['duration_ms'].mean() / 60000
    features['song_length_preference'] = min(1.0, avg_duration_min / 5)
    features['short_song_preference'] = (track_df['duration_ms'] < 180000).mean()
    features['long_song_preference'] = (track_df['duration_ms'] > 300000).mean()
    
    # Genre diversity
    genre_diversity = len(set(all_genres)) / max(len(all_genres), 1) if all_genres else 0
    features['genre_diversity'] = genre_diversity
    features['unique_genres_count'] = len(set(all_genres))
    
    st.write(f"🐛 DEBUG: Genre diversity: {genre_diversity:.3f}")
    
    # Analyze genres for personality indicators with detailed logging
    electronic_terms = ['electronic', 'dance', 'house', 'techno', 'edm', 'club', 'party']
    electronic_count = sum(1 for term in electronic_terms if term in genre_text)
    features['electronic_preference'] = electronic_count / max(len(all_genres), 1)
    
    rock_terms = ['rock', 'metal', 'punk', 'alternative', 'grunge', 'indie']
    rock_count = sum(1 for term in rock_terms if term in genre_text)
    features['rock_preference'] = rock_count / max(len(all_genres), 1)
    
    calm_terms = ['acoustic', 'folk', 'ambient', 'chill', 'soft', 'mellow', 'classical']
    calm_count = sum(1 for term in calm_terms if term in genre_text)
    features['calm_preference'] = calm_count / max(len(all_genres), 1)
    
    pop_terms = ['pop', 'mainstream', 'chart', 'radio']
    pop_count = sum(1 for term in pop_terms if term in genre_text)
    features['pop_preference'] = pop_count / max(len(all_genres), 1)
    
    st.write(f"🐛 DEBUG: Genre preferences:")
    st.write(f"  - Electronic: {features['electronic_preference']:.3f} (found {electronic_count} matches)")
    st.write(f"  - Rock: {features['rock_preference']:.3f} (found {rock_count} matches)")
    st.write(f"  - Calm: {features['calm_preference']:.3f} (found {calm_count} matches)")
    st.write(f"  - Pop: {features['pop_preference']:.3f} (found {pop_count} matches)")
    
    # Artist diversity
    unique_artists = len(set([track['artists'][0]['name'] for track in tracks]))
    features['artist_diversity'] = unique_artists / len(tracks)
    
    # Create personality indicators
    features['energy_preference'] = (
        features['electronic_preference'] * 0.4 +
        features['rock_preference'] * 0.4 +
        features['pop_preference'] * 0.2
    )
    
    features['openness_indicators'] = (
        features['genre_diversity'] * 0.4 +
        features['underground_preference'] * 0.3 +
        features['artist_diversity'] * 0.3
    )
    
    features['conscientiousness_indicators'] = (
        features['mainstream_preference'] * 0.5 +
        features['pop_preference'] * 0.3 +
        (1 - features['explicit_content_ratio']) * 0.2
    )
    
    st.write(f"🐛 DEBUG: Personality indicators:")
    st.write(f"  - Energy: {features['energy_preference']:.3f}")
    st.write(f"  - Openness: {features['openness_indicators']:.3f}")
    st.write(f"  - Conscientiousness: {features['conscientiousness_indicators']:.3f}")
    
    # Ensure all values are in [0, 1] range and handle edge cases
    for key, value in features.items():
        if hasattr(value, 'item'):
            value = value.item()
        elif hasattr(value, '__len__') and not isinstance(value, str):
            value = float(value.mean() if hasattr(value, 'mean') else value[0])
        
        if pd.isna(value) or value is None:
            st.write(f"🐛 DEBUG: {key} was NaN/None, setting to 0.5")
            features[key] = 0.5
        else:
            features[key] = max(0.0, min(1.0, float(value)))
    
    # Store features for comparison
    st.session_state.debug_data['features'] = features
    st.session_state.debug_data['feature_summary'] = {
        'user_id': user_id,
        'track_count': len(tracks),
        'unique_artists': unique_artists,
        'genre_count': len(all_genres),
        'unique_genres': len(set(all_genres)),
        'avg_popularity': track_df['popularity'].mean(),
        'key_features': {
            'energy_preference': features['energy_preference'],
            'openness_indicators': features['openness_indicators'],
            'mainstream_preference': features['mainstream_preference']
        }
    }
    
    st.write(f"🐛 DEBUG: Final feature count: {len(features)}")
    st.write(f"🐛 DEBUG: Feature keys: {list(features.keys())}")
    
    return features

def predict_personality_debug(features, loaded_models):
    """DEBUG: Predict personality with detailed logging"""
    
    st.write("🐛 DEBUG: Starting personality prediction...")
    
    if not features:
        st.error("🐛 DEBUG: No features provided!")
        return None, None
    
    st.write(f"🐛 DEBUG: Input features count: {len(features)}")
    st.write(f"🐛 DEBUG: Sample features: {dict(list(features.items())[:5])}")
    
    # Try models in order with detailed error reporting
    for model_name in ['behavioral', 'research', 'per']:
        if model_name in loaded_models:
            st.write(f"🐛 DEBUG: Trying {model_name} model...")
            
            try:
                model_data = loaded_models[model_name]
                
                if model_name == 'behavioral':
                    # Check feature count mismatch
                    expected_features = model_data.get('feature_lists', {})
                    if expected_features:
                        first_trait = list(expected_features.keys())[0]
                        expected_count = len(expected_features[first_trait])
                        st.write(f"🐛 DEBUG: Behavioral model expects {expected_count} features, we have {len(features)}")
                
                elif model_name == 'research':
                    # Try research model
                    models = model_data.get('models', {})
                    scalers = model_data.get('scalers', {})
                    feature_cols = model_data.get('feature_cols', [])
                    
                    st.write(f"🐛 DEBUG: Research model has {len(models)} trait models")
                    st.write(f"🐛 DEBUG: Expected feature columns: {feature_cols}")
                    
                    if not models:
                        st.write("🐛 DEBUG: No models found in research data")
                        continue
                    
                    # Create feature vector
                    feature_vector = []
                    missing_features = []
                    
                    for col in feature_cols:
                        if col in features:
                            feature_vector.append(features[col])
                        else:
                            feature_vector.append(0.5)  # Default
                            missing_features.append(col)
                    
                    if missing_features:
                        st.write(f"🐛 DEBUG: Missing features (using defaults): {missing_features}")
                    
                    feature_array = np.array(feature_vector).reshape(1, -1)
                    st.write(f"🐛 DEBUG: Feature array shape: {feature_array.shape}")
                    st.write(f"🐛 DEBUG: Feature array sample: {feature_array[0][:5]}")
                    
                    predictions = {}
                    confidence = {}
                    
                    for trait in models:
                        try:
                            scaler = scalers.get(trait)
                            if scaler:
                                scaled_features = scaler.transform(feature_array)
                                st.write(f"🐛 DEBUG: {trait} - scaled features sample: {scaled_features[0][:3]}")
                            else:
                                scaled_features = feature_array
                                st.write(f"🐛 DEBUG: {trait} - no scaler, using raw features")
                            
                            prediction = models[trait].predict(scaled_features)[0]
                            clipped_prediction = np.clip(prediction, 1.0, 5.0)
                            predictions[trait] = round(clipped_prediction, 2)
                            confidence[trait] = 0.7
                            
                            st.write(f"🐛 DEBUG: {trait} prediction: {prediction:.3f} -> {clipped_prediction:.2f}")
                            
                        except Exception as e:
                            st.write(f"🐛 DEBUG: Error predicting {trait}: {e}")
                            predictions[trait] = 3.0
                            confidence[trait] = 0.1
                    
                    if predictions:
                        st.write(f"🐛 DEBUG: Final predictions: {predictions}")
                        st.session_state.debug_data['predictions'] = predictions
                        st.session_state.debug_data['model_used'] = model_name
                        return predictions, confidence
                
            except Exception as e:
                st.write(f"🐛 DEBUG: Error with {model_name} model: {e}")
                import traceback
                st.write(f"🐛 DEBUG: Traceback: {traceback.format_exc()}")
                continue
    
    # Fallback prediction
    st.write("🐛 DEBUG: Using fallback prediction...")
    fallback_predictions = {
        'Extraversion': 3.0,
        'Openness': 3.0,
        'Conscientiousness': 3.0,
        'Agreeableness': 3.0,
        'Neuroticism': 3.0
    }
    
    st.session_state.debug_data['predictions'] = fallback_predictions
    st.session_state.debug_data['model_used'] = 'fallback'
    
    return fallback_predictions, {'Extraversion': 0.1, 'Openness': 0.1, 'Conscientiousness': 0.1, 'Agreeableness': 0.1, 'Neuroticism': 0.1}

def main():
    st.title("🐛 Spotify Personality Predictor - DEBUG VERSION")
    st.markdown("### Debug mode to identify why all users get identical results")
    
    # Load models
    loaded_models = load_trained_models()
    
    # Authentication
    sp = ensure_spotify_client()
    st.success("Connected to Spotify")
    
    # Show debug data from previous sessions
    if st.session_state.debug_data:
        with st.expander("🐛 Previous Session Data"):
            st.json(st.session_state.debug_data)
    
    if st.button("🐛 DEBUG: Analyze My Musical Personality", type="primary"):
        
        # Clear previous debug data
        st.session_state.debug_data = {}
        
        with st.spinner("🐛 DEBUG: Collecting music data..."):
            music_data = get_user_music_data_debug(sp)
        
        if music_data and music_data['tracks']:
            
            with st.spinner("🐛 DEBUG: Extracting features..."):
                features = create_features_from_metadata_debug(music_data)
            
            if features:
                with st.spinner("🐛 DEBUG: Predicting personality..."):
                    predictions, confidence = predict_personality_debug(features, loaded_models)
                
                if predictions:
                    # Show debug summary
                    st.header("🐛 DEBUG SUMMARY")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("User Data")
                        st.write(f"User ID: {music_data.get('user_id', 'unknown')}")
                        st.write(f"Tracks: {len(music_data['tracks'])}")
                        st.write(f"Artists: {len(set(music_data['artist_names']))}")
                        st.write(f"Genres: {len(music_data['genres'])}")
                        
                    with col2:
                        st.subheader("Key Features")
                        if 'feature_summary' in st.session_state.debug_data:
                            summary = st.session_state.debug_data['feature_summary']
                            for key, value in summary['key_features'].items():
                                st.write(f"{key}: {value:.3f}")
                    
                    # Show predictions
                    st.header("Personality Results")
                    
                    for trait, score in predictions.items():
                        conf = confidence.get(trait, 0.5)
                        st.metric(trait, f"{score:.1f}/5.0", f"Confidence: {conf:.0%}")
                    
                    # Show all extracted features for debugging
                    with st.expander("🐛 All Extracted Features"):
                        features_df = pd.DataFrame([features]).T
                        features_df.columns = ['Value']
                        st.dataframe(features_df)
                    
                    # Compare with previous users
                    st.header("🐛 User Comparison")
                    st.write("Compare this analysis with previous users to identify if results are identical")
                    
                    if 'debug_data' in st.session_state:
                        st.json(st.session_state.debug_data)
        
        else:
            st.error("🐛 DEBUG: Could not get music data")

if __name__ == "__main__":
    main()