# app.py - Fixed Spotify Personality Predictor with Proper User Isolation
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
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

if LASTFM_API_KEY:
    st.sidebar.success("✅ Last.fm integration enabled")
else:
    st.sidebar.info("ℹ️ Last.fm API not configured")

# -----------------------------------------------------------------------------
# FIXED Authentication with Session-Specific Cache Files
# -----------------------------------------------------------------------------
def ensure_spotify_client():
    """Fixed authentication using session-specific cache files"""
    
    # Generate unique session ID for this user session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(int(time.time() * 1000))

    session_id = st.session_state.session_id
    cache_path = f".cache-{CLIENT_ID}-{session_id}"

    # Check for authorization code in URL
    params = st.query_params
    if "code" in params:
        code = params["code"]
        
        auth_manager = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=cache_path,
            show_dialog=True
        )

        try:
            token_info = auth_manager.get_access_token(code, as_dict=True)
            if token_info and "access_token" in token_info:
                sp = spotipy.Spotify(auth=token_info["access_token"])
                
                # Immediately verify user identity
                user = sp.current_user()
                st.success(f"Authenticated as: {user.get('display_name', 'Unknown')} ({user.get('id', 'Unknown')})")
                
                st.query_params.clear()

                # Save to session state
                st.session_state.sp = sp
                st.session_state.user = user
                st.session_state.token_info = token_info

                st.rerun()
            else:
                st.error("No access token received")
                return None
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return None

    # Check if we already have authenticated client in session
    if hasattr(st.session_state, 'sp') and st.session_state.sp:
        return st.session_state.sp

    # Need to authenticate
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=cache_path,
        show_dialog=True,
        state=f"spotify_{session_id}"
    )

    login_url = auth_manager.get_authorize_url()
    st.info("Please log in with Spotify")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

# -----------------------------------------------------------------------------
# Model Loading and Management
# -----------------------------------------------------------------------------
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
    
    if not loaded_models:
        st.sidebar.info("ℹ️ No trained models found - using fallback prediction")
    
    return loaded_models

def create_model_features_from_spotify_data(music_data):
    """Create features that match what the trained models expect"""
    
    audio_features = music_data['audio_features']
    genres = music_data['genres']
    tracks = music_data['tracks']
    
    if not audio_features:
        return None
    
    df = pd.DataFrame(audio_features)
    
    # Create features that match the behavioral model training
    features = {
        # Basic audio features (matching training data)
        'energy_preference': df['energy'].mean(),
        'social_music_score': (df['valence'] + df['danceability']).mean() / 2,
        'high_energy_preference': (df['energy'] > 0.7).astype(int).mean(),
        'danceable_preference': df['danceability'].mean(),
        'loudness_preference': (df['loudness'] + 60) / 60,  # Normalize to 0-1
        'tempo_preference': df['tempo'].mean() / 200,  # Normalize
        
        # Openness features
        'musical_complexity': (df['acousticness'] + df['instrumentalness']).mean() / 2,
        'experimental_preference': df['instrumentalness'].mean(),
        'acoustic_exploration': df['acousticness'].mean(),
        'instrumental_preference': df['instrumentalness'].mean(),
        'genre_openness': len(set(genres)) / max(len(genres), 1),
        'unconventional_preference': 1 - (df['popularity'].mean() / 100),
        
        # Conscientiousness features
        'listening_consistency': 1 - df['valence'].std() if len(df) > 1 else 0.8,
        'routine_preference': df['popularity'].mean() / 100,
        'completion_tendency': df['energy'].mean(),  # Proxy for not skipping
        'organized_listening': (df['popularity'] > 50).mean(),
        'mainstream_preference': (df['popularity'] > 70).mean(),
        'predictable_choice': 1 - df['energy'].std() if len(df) > 1 else 0.8,
        
        # Agreeableness features
        'positive_music_preference': df['valence'].mean(),
        'mellow_preference': 1 - df['energy'].mean(),
        'harmony_seeking': df['valence'].mean(),
        'avoid_aggressive': 1 - (df['energy'] * (1 - df['valence'])).mean(),
        'social_acceptance': (df['valence'] + df['danceability']).mean() / 2,
        'cooperative_music': df['acousticness'].mean(),
        
        # Neuroticism features
        'emotional_music_seeking': 1 - df['valence'].mean(),
        'mood_instability': df['valence'].std() if len(df) > 1 else 0.1,
        'anxiety_music': ((1 - df['valence']) * df['energy']).mean(),
        'comfort_seeking': df['acousticness'].mean(),
        'emotional_volatility': df['energy'].std() if len(df) > 1 else 0.1,
        'stress_response': (1 - df['valence']).mean(),
        
        # General features
        'music_sophistication': (df['acousticness'] + df['instrumentalness']).mean() / 2,
        'emotional_regulation': df['valence'].mean(),
        'stimulation_seeking': df['energy'].mean(),
        'mood_management': df['valence'].mean(),
        'musical_engagement': df['danceability'].mean(),
    }
    
    # Ensure all values are in [0, 1] range
    for key, value in features.items():
        # Handle pandas Series or numpy arrays
        if hasattr(value, 'item'):  # pandas Series or numpy scalar
            value = value.item() if hasattr(value, 'item') else float(value)
        elif hasattr(value, '__len__') and not isinstance(value, str):  # Handle arrays/series
            value = float(value.mean() if hasattr(value, 'mean') else value[0])
        
        if pd.isna(value) or value is None:
            features[key] = 0.5
        else:
            features[key] = max(0.0, min(1.0, float(value)))
    
    return features

# -----------------------------------------------------------------------------
# Last.fm Integration for Data Enrichment
# -----------------------------------------------------------------------------
def get_lastfm_artist_info(artist_name):
    """Get additional artist info from Last.fm using your API key"""
    if not LASTFM_API_KEY:
        return None
    
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'artist.getInfo',
            'artist': artist_name,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'artist' in data:
                artist_info = data['artist']
                return {
                    'tags': [tag['name'] for tag in artist_info.get('tags', {}).get('tag', [])],
                    'listeners': int(artist_info.get('stats', {}).get('listeners', 0)),
                    'playcount': int(artist_info.get('stats', {}).get('playcount', 0)),
                    'bio_summary': artist_info.get('bio', {}).get('summary', ''),
                    'similar_artists': [artist['name'] for artist in artist_info.get('similar', {}).get('artist', [])]
                }
    except Exception as e:
        print(f"Last.fm API error for {artist_name}: {e}")
    return None

def enrich_with_musicbrainz(artist_name):
    """Get genre info from MusicBrainz (free, no API key needed)"""
    try:
        # Search for artist
        search_url = f"https://musicbrainz.org/ws/2/artist/?query={artist_name}&fmt=json&limit=1"
        headers = {'User-Agent': 'SpotifyPersonalityApp/1.0'}
        
        response = requests.get(search_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            artists = data.get('artists', [])
            if artists:
                artist_id = artists[0]['id']
                
                # Get detailed artist info
                detail_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=tags&fmt=json"
                detail_response = requests.get(detail_url, headers=headers, timeout=5)
                
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    tags = detail_data.get('tags', [])
                    return [tag['name'] for tag in tags if tag.get('count', 0) > 0]
        
        time.sleep(1)  # Rate limiting
    except:
        pass
    return []

# -----------------------------------------------------------------------------
# Music Data Collection with Proper User Isolation
# -----------------------------------------------------------------------------
def get_user_music_data(sp, limit=50):
    """Get user's music data with proper isolation"""
    
    music_data = {
        'tracks': [],
        'artists': [],
        'genres': [],
        'audio_features': [],
        'enriched_data': {}
    }
    
    try:
        st.write("🎵 Fetching your music library...")
        
        all_tracks = []
        
        # Get tracks from multiple sources
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                tracks = sp.current_user_top_tracks(limit=limit, time_range=time_range)
                all_tracks.extend(tracks['items'])
                st.write(f"✅ Got {len(tracks['items'])} tracks from {time_range}")
            except Exception as e:
                st.warning(f"Could not get {time_range} tracks: {e}")
        
        # Get recent tracks
        try:
            recent = sp.current_user_recently_played(limit=limit)
            recent_tracks = [item['track'] for item in recent['items']]
            all_tracks.extend(recent_tracks)
            st.write(f"✅ Got {len(recent_tracks)} recent tracks")
        except Exception as e:
            st.warning(f"Could not get recent tracks: {e}")
        
        if not all_tracks:
            st.error("No tracks found. Make sure you have listening history!")
            return None
        
        # Remove duplicates
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track['id'] and track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        music_data['tracks'] = unique_tracks
        st.success(f"📊 Found {len(unique_tracks)} unique tracks")
        
        # Get audio features
        if unique_tracks:
            try:
                track_ids = [track['id'] for track in unique_tracks]
                audio_features = []
                
                # Process in batches to avoid rate limits
                batch_size = 50
                for i in range(0, len(track_ids), batch_size):
                    batch_ids = track_ids[i:i+batch_size]
                    try:
                        features = sp.audio_features(batch_ids)
                        audio_features.extend([f for f in features if f is not None])
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        st.warning(f"Could not get audio features for batch {i//batch_size + 1}: {e}")
                
                music_data['audio_features'] = audio_features
                st.write(f"🎵 Got audio features for {len(audio_features)} tracks")
                
            except Exception as e:
                st.warning(f"Could not get audio features: {e}")
        
        # Get artist info for genres
        artists_info = get_artist_info_robust(sp, unique_tracks)
        music_data['artists'] = artists_info['artists']
        music_data['genres'] = artists_info['genres']
        
        # Enrich with Last.fm if available
        if LASTFM_API_KEY:
            st.write("🌍 Enriching with Last.fm data...")
            enriched_data = enrich_with_lastfm(unique_tracks)
            music_data['enriched_data'] = enriched_data
        
        return music_data
        
    except Exception as e:
        st.error(f"Error collecting music data: {e}")
        return None

def get_artist_info_robust(sp, tracks):
    """Get artist info with error handling"""
    
    st.write("🎤 Getting artist information...")
    
    artist_ids = list(set([track['artists'][0]['id'] for track in tracks if track['artists']]))
    all_artists = []
    all_genres = []
    
    for i in range(0, len(artist_ids), 20):
        batch_ids = artist_ids[i:i+20]
        try:
            artists_response = sp.artists(batch_ids)
            batch_artists = artists_response['artists']
            all_artists.extend(batch_artists)
            
            for artist in batch_artists:
                all_genres.extend(artist.get('genres', []))
            
            time.sleep(0.3)
            
        except Exception as e:
            st.warning(f"Error getting artist info: {e}")
            for track in tracks:
                if track['artists'][0]['id'] in batch_ids:
                    all_genres.append('unknown')
    
    return {'artists': all_artists, 'genres': all_genres}

def enrich_with_lastfm(tracks):
    """Enrich track data using Last.fm"""
    enriched_data = {
        'artist_tags': {},
        'track_tags': {},
        'artist_popularity': {},
        'genres_enriched': []
    }
    
    # Process a sample of tracks to avoid API limits
    sample_tracks = tracks[:20]  # Limit to avoid too many API calls
    
    for i, track in enumerate(sample_tracks):
        if i % 5 == 0:
            st.write(f"   Enriching track {i+1}/{len(sample_tracks)}")
        
        artist_name = track['artists'][0]['name']
        track_name = track['name']
        
        # Get artist info
        if artist_name not in enriched_data['artist_tags']:
            artist_info = get_lastfm_artist_info(artist_name)
            if artist_info:
                enriched_data['artist_tags'][artist_name] = artist_info.get('tags', [])
                enriched_data['artist_popularity'][artist_name] = artist_info.get('listeners', 0)
                enriched_data['genres_enriched'].extend(artist_info.get('tags', []))
        
        time.sleep(0.2)  # Rate limiting
    
    return enriched_data

# -----------------------------------------------------------------------------
# Model-Based Personality Prediction (same as original)
# -----------------------------------------------------------------------------
def predict_personality_with_models(features, loaded_models):
    """Use trained models to predict personality"""
    
    if not loaded_models or not features:
        return predict_personality_fallback(features)
    
    predictions = {}
    prediction_confidence = {}
    
    # Try to use the best available model
    model_priority = ['behavioral', 'research', 'per']
    
    for model_name in model_priority:
        if model_name in loaded_models:
            try:
                model_data = loaded_models[model_name]
                
                if model_name == 'behavioral':
                    predictions, confidence = predict_with_behavioral_model(features, model_data)
                elif model_name == 'research':
                    predictions, confidence = predict_with_research_model(features, model_data)
                elif model_name == 'per':
                    predictions, confidence = predict_with_per_model(features, model_data)
                
                if predictions:
                    st.success(f"✅ Used {model_name} model for prediction")
                    return predictions, confidence
                    
            except Exception as e:
                st.warning(f"⚠️ Error using {model_name} model: {e}")
                continue
    
    # Fallback to rule-based if all models fail
    st.info("ℹ️ Using fallback rule-based prediction")
    predictions = predict_personality_fallback(features)
    confidence = {trait: 0.6 for trait in predictions.keys()}
    return predictions, confidence

def predict_with_behavioral_model(features, model_data):
    """Use the behavioral model we trained"""
    try:
        models = model_data['models']
        scalers = model_data['scalers']
        selectors = model_data.get('selectors', {})
        feature_lists = model_data.get('feature_lists', {})
        
        predictions = {}
        confidence = {}
        
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            if trait in models:
                # Get the features this model expects
                expected_features = feature_lists.get(trait, list(features.keys()))
                
                # Create feature vector
                feature_vector = []
                for feature_name in expected_features:
                    if feature_name in features:
                        feature_vector.append(features[feature_name])
                    else:
                        feature_vector.append(0.5)
                
                feature_array = np.array(feature_vector).reshape(1, -1)
                
                # Scale features
                if trait in scalers:
                    scaled_features = scalers[trait].transform(feature_array)
                else:
                    scaled_features = feature_array
                
                # Apply feature selection if available
                if trait in selectors and selectors[trait] is not None:
                    scaled_features = selectors[trait].transform(scaled_features)
                
                # Predict
                prediction = models[trait].predict(scaled_features)[0]
                
                # Clip to valid range
                prediction = np.clip(prediction, 1.0, 5.0)
                predictions[trait] = round(prediction, 2)
                
                # Calculate confidence
                perf = model_data.get('performance_results', {}).get(trait, {})
                model_r2 = perf.get('test_r2', 0.0)
                confidence[trait] = max(0.1, min(0.9, model_r2 + 0.5))
        
        return predictions, confidence
        
    except Exception as e:
        st.error(f"Error in behavioral model prediction: {e}")
        return None, None

def predict_with_research_model(features, model_data):
    """Use research-based model"""
    try:
        models = model_data.get('models', {})
        scalers = model_data.get('scalers', {})
        feature_cols = model_data.get('feature_cols', list(features.keys()))
        
        predictions = {}
        confidence = {}
        
        # Create feature vector
        feature_vector = []
        for col in feature_cols:
            if col in features:
                feature_vector.append(features[col])
            else:
                feature_vector.append(0.5)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        for trait in models:
            scaler = scalers.get(trait)
            if scaler:
                scaled_features = scaler.transform(feature_array)
            else:
                scaled_features = feature_array
            
            prediction = models[trait].predict(scaled_features)[0]
            predictions[trait] = round(np.clip(prediction, 1.0, 5.0), 2)
            confidence[trait] = 0.7
        
        return predictions, confidence
        
    except Exception as e:
        st.error(f"Error in research model prediction: {e}")
        return None, None

def predict_with_per_model(features, model_data):
    """Use PER dataset model"""
    try:
        models = model_data.get('models', {})
        feature_names = model_data.get('features', list(features.keys()))
        
        predictions = {}
        confidence = {}
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.5)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        for trait in models:
            prediction = models[trait].predict(feature_array)[0]
            predictions[trait] = round(np.clip(prediction, 1.0, 5.0), 2)
            confidence[trait] = 0.3
        
        return predictions, confidence
        
    except Exception as e:
        st.error(f"Error in PER model prediction: {e}")
        return None, None

def predict_personality_fallback(features):
    """Fallback rule-based prediction when models fail"""
    
    if not features:
        return {trait: 3.0 for trait in ['Extraversion', 'Openness', 'Conscientiousness', 'Agreeableness', 'Neuroticism']}
    
    predictions = {}
    
    # EXTRAVERSION
    extraversion = (
        features.get('energy_preference', 0.5) * 0.3 +
        features.get('social_music_score', 0.5) * 0.25 +
        features.get('danceable_preference', 0.5) * 0.2 +
        features.get('high_energy_preference', 0.3) * 0.15 +
        features.get('mainstream_preference', 0.3) * 0.1
    )
    predictions['Extraversion'] = np.clip(extraversion * 5, 1, 5)
    
    # OPENNESS
    openness = (
        features.get('genre_openness', 0.3) * 0.3 +
        features.get('experimental_preference', 0.1) * 0.2 +
        features.get('unconventional_preference', 0.3) * 0.2 +
        features.get('musical_complexity', 0.3) * 0.15 +
        features.get('acoustic_exploration', 0.3) * 0.15
    )
    predictions['Openness'] = np.clip(openness * 5, 1, 5)
    
    # CONSCIENTIOUSNESS
    conscientiousness = (
        features.get('mainstream_preference', 0.3) * 0.3 +
        features.get('listening_consistency', 0.8) * 0.25 +
        features.get('routine_preference', 0.5) * 0.2 +
        features.get('organized_listening', 0.5) * 0.15 +
        features.get('predictable_choice', 0.8) * 0.1
    )
    predictions['Conscientiousness'] = np.clip(conscientiousness * 5, 1, 5)
    
    # AGREEABLENESS
    agreeableness = (
        features.get('positive_music_preference', 0.5) * 0.35 +
        features.get('harmony_seeking', 0.5) * 0.25 +
        features.get('social_acceptance', 0.5) * 0.2 +
        features.get('avoid_aggressive', 0.7) * 0.2
    )
    predictions['Agreeableness'] = np.clip(agreeableness * 5, 1, 5)
    
    # NEUROTICISM
    neuroticism = (
        features.get('emotional_music_seeking', 0.5) * 0.3 +
        features.get('mood_instability', 0.1) * 0.25 +
        features.get('anxiety_music', 0.3) * 0.2 +
        features.get('emotional_volatility', 0.1) * 0.15 +
        features.get('stress_response', 0.3) * 0.1
    )
    predictions['Neuroticism'] = np.clip(neuroticism * 5, 1, 5)
    
    # Round results
    for trait in predictions:
        predictions[trait] = round(predictions[trait], 2)
    
    return predictions

# -----------------------------------------------------------------------------
# UI and Insights (same as original)
# -----------------------------------------------------------------------------
def create_personality_insights(predictions, confidence=None):
    """Generate personality insights with confidence indicators"""
    
    insights = {}
    
    descriptions = {
        'Extraversion': {
            'high': "You love energetic, social music that gets people moving!",
            'medium': "You balance energetic and calm music well.",
            'low': "You prefer introspective, quieter music for personal reflection."
        },
        'Openness': {
            'high': "You're a musical explorer who loves discovering new sounds!",
            'medium': "You balance familiar and new music nicely.",
            'low': "You stick to what you know and love - reliable taste!"
        },
        'Conscientiousness': {
            'high': "Your music habits are organized and consistent.",
            'medium': "You have some structure but stay flexible.",
            'low': "Your music choices are spontaneous and mood-driven!"
        },
        'Agreeableness': {
            'high': "You love positive, harmonious music that brings people together!",
            'medium': "You appreciate both uplifting and complex music.",
            'low': "You're drawn to more intense, unconventional sounds."
        },
        'Neuroticism': {
            'high': "Music is your emotional outlet and processing tool.",
            'medium': "Your music reflects your varied emotional states.",
            'low': "You prefer stable, positive music for good vibes."
        }
    }
    
    for trait, score in predictions.items():
        if score >= 3.5:
            category = 'high'
        elif score <= 2.5:
            category = 'low'
        else:
            category = 'medium'
        
        trait_confidence = confidence.get(trait, 0.6) if confidence else 0.6
        
        insights[trait] = {
            'score': score,
            'description': descriptions[trait][category],
            'confidence': trait_confidence
        }
    
    return insights

# -----------------------------------------------------------------------------
# Main App with Fixed Authentication
# -----------------------------------------------------------------------------
def main():
    st.title("Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits from your music!")
    
    # Show session info in sidebar for debugging
    if "session_id" in st.session_state:
        st.sidebar.write(f"Session: {st.session_state.session_id[:8]}...")
    
    # Load trained models
    loaded_models = load_trained_models()
    
    # Fixed authentication
    sp = ensure_spotify_client()
    
    # Show authenticated user
    if hasattr(st.session_state, 'user') and st.session_state.user:
        user = st.session_state.user
        st.success(f"Logged in as: {user.get('display_name', 'Unknown')} ({user.get('id', 'Unknown')})")

    if st.button("Analyze My Musical Personality", type="primary"):
        
        with st.spinner("Collecting and analyzing your music data..."):
            music_data = get_user_music_data(sp)
        
        if music_data and music_data['tracks']:
            
            # Extract features
            features = create_model_features_from_spotify_data(music_data)
            
            if features:
                # Use trained models for prediction
                with st.spinner("Running personality prediction models..."):
                    prediction_result = predict_personality_with_models(features, loaded_models)
                
                if isinstance(prediction_result, tuple):
                    predictions, confidence = prediction_result
                else:
                    predictions = prediction_result
                    confidence = None
                
                insights = create_personality_insights(predictions, confidence)
                
                # Display results
                st.header("Your Musical Personality")
                
                # Show analysis method
                st.info("Analysis based on your unique Spotify data using trained ML models")
                
                # Create radar chart
                fig = go.Figure()
                
                traits = list(predictions.keys())
                scores = list(predictions.values())
                
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=traits,
                    fill='toself',
                    name='Your Personality'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                    showlegend=False,
                    title="Big Five Personality Traits (1-5 scale)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show insights with confidence
                for trait, insight in insights.items():
                    conf_text = ""
                    if confidence and trait in confidence:
                        conf_percent = confidence[trait] * 100
                        conf_text = f" (Confidence: {conf_percent:.0f}%)"
                    
                    with st.expander(f"{trait}: {insight['score']:.1f}/5.0{conf_text}"):
                        st.write(insight['description'])
                        if confidence and trait in confidence:
                            st.progress(confidence[trait])
                
                # Show music analysis
                st.subheader("Your Music Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tracks Analyzed", len(music_data['tracks']))
                    if features.get('mainstream_preference'):
                        st.metric("Mainstream Score", f"{features['mainstream_preference']:.0%}")
                
                with col2:
                    st.metric("Unique Genres", len(set(music_data['genres'])))
                    if features.get('genre_openness'):
                        st.metric("Genre Diversity", f"{features['genre_openness']:.2f}")
                
                with col3:
                    unique_artists = len(set([track['artists'][0]['name'] for track in music_data['tracks']]))
                    st.metric("Unique Artists", unique_artists)
                    if music_data['audio_features']:
                        avg_energy = np.mean([f['energy'] for f in music_data['audio_features'] if f])
                        st.metric("Avg Energy", f"{avg_energy:.2f}")
                
                # Audio features visualization
                if music_data['audio_features']:
                    st.subheader("Your Audio Profile")
                    
                    audio_df = pd.DataFrame(music_data['audio_features'])
                    audio_means = {
                        'Energy': audio_df['energy'].mean(),
                        'Danceability': audio_df['danceability'].mean(),
                        'Valence': audio_df['valence'].mean(),
                        'Acousticness': audio_df['acousticness'].mean(),
                        'Instrumentalness': audio_df['instrumentalness'].mean()
                    }
                    
                    audio_chart_df = pd.DataFrame(list(audio_means.items()), columns=['Feature', 'Score'])
                    st.bar_chart(audio_chart_df.set_index('Feature'))
                
                # Genre preferences
                if music_data['genres']:
                    st.subheader("Your Top Genres")
                    genre_counts = Counter(music_data['genres']).most_common(10)
                    genre_df = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
                    st.bar_chart(genre_df.set_index('Genre'))
                
                # Model info
                if loaded_models:
                    with st.expander("Model Information"):
                        st.write(f"Using {len(loaded_models)} trained model(s) for prediction")
                        for model_name in loaded_models.keys():
                            st.write(f"- {model_name.title()} Model")
                
                # User verification (for debugging)
                if hasattr(st.session_state, 'user'):
                    with st.expander("User Verification"):
                        user = st.session_state.user
                        st.write(f"Analysis for: {user.get('display_name', 'Unknown')} ({user.get('id', 'Unknown')})")
                        st.write(f"Session ID: {st.session_state.session_id}")
        
        else:
            st.error("Could not analyze your music. This might be due to API limitations or insufficient listening history.")
            st.info("Try again later, or make sure you have recent listening activity on Spotify.")

if __name__ == "__main__":
    main()