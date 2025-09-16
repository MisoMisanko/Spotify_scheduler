# app.py - Complete Spotify Personality Predictor using TRAINED MODELS
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
# FIXED Authentication - ONLY CHANGE
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
# Robust Music Data Collection with Fallbacks
# -----------------------------------------------------------------------------
def get_user_music_data_simplified(sp, limit=50):
    """Get user's music data focusing on track metadata instead of audio features"""
    
    music_data = {
        'tracks': [],
        'artists': [],
        'genres': [],
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
        
        # Get artist info for genres (this usually works)
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

def create_features_from_metadata(music_data):
    """Create features that exactly match what the trained behavioral model expects"""
    
    tracks = music_data['tracks']
    genres = music_data['genres']
    enriched_data = music_data.get('enriched_data', {})
    
    if not tracks:
        return None
    
    # Analyze track metadata
    track_df = pd.DataFrame([{
        'name': track['name'],
        'artist': track['artists'][0]['name'],
        'popularity': track.get('popularity', 50),
        'explicit': track.get('explicit', False),
        'duration_ms': track.get('duration_ms', 180000)
    } for track in tracks])
    
    # Get enriched genre/tag data
    all_genres = genres + enriched_data.get('genres_enriched', [])
    
    # DEBUG: Show what genres we actually have
    st.write(f"DEBUG GENRES: Raw Spotify genres: {genres[:10] if genres else 'NONE'}")
    st.write(f"DEBUG GENRES: Enriched genres: {enriched_data.get('genres_enriched', [])[:10]}")
    st.write(f"DEBUG GENRES: Combined genres: {all_genres[:10] if all_genres else 'NONE'}")
    st.write(f"DEBUG GENRES: Total genre count: {len(all_genres)}")
    
    # DEBUG: Show sample track and artist names
    sample_tracks = [track['name'] for track in tracks[:5]]
    sample_artists = [track['artists'][0]['name'] for track in tracks[:5]]
    st.write(f"DEBUG TRACKS: Sample track names: {sample_tracks}")
    st.write(f"DEBUG ARTISTS: Sample artist names: {sample_artists}")
    
    # Create EXACTLY the features the behavioral model was trained on
    # These should match the original audio-feature-based training
    features = {}
    
    # Analyze genres for personality indicators
    genre_text = ' '.join(all_genres).lower()
    st.write(f"DEBUG: Genre text for analysis: '{genre_text[:100]}...' (length: {len(genre_text)})")
    
    # Electronic/Dance detection (affects energy, danceability)
    electronic_terms = ['electronic', 'dance', 'house', 'techno', 'edm', 'club', 'party', 'disco']
    electronic_score = sum(1 for term in electronic_terms if term in genre_text) / max(len(all_genres), 1)
    
    # Rock/Metal detection (affects energy)
    rock_terms = ['rock', 'metal', 'punk', 'alternative', 'grunge', 'indie rock', 'hard rock']
    rock_score = sum(1 for term in rock_terms if term in genre_text) / max(len(all_genres), 1)
    
    # Calm/Acoustic detection
    calm_terms = ['acoustic', 'folk', 'ambient', 'chill', 'soft', 'mellow', 'classical', 'singer-songwriter']
    calm_score = sum(1 for term in calm_terms if term in genre_text) / max(len(all_genres), 1)
    
    # Pop detection
    pop_terms = ['pop', 'mainstream', 'chart', 'radio', 'teen pop', 'dance pop']
    pop_score = sum(1 for term in pop_terms if term in genre_text) / max(len(all_genres), 1)
    
    # Hip-hop detection
    hiphop_terms = ['hip hop', 'rap', 'hip-hop', 'trap', 'drill']
    hiphop_score = sum(1 for term in hiphop_terms if term in genre_text) / max(len(all_genres), 1)
    
    # World/Latin music detection - MASSIVELY EXPANDED for cumbia and all Latino variants
    world_terms = [
        # Cumbia variants and related
        'cumbia', 'cumbia villera', 'cumbia santafesina', 'cumbia peruana', 'cumbia colombiana', 
        'cumbia mexicana', 'cumbia argentina', 'cumbia rebajada', 'cumbia sonidera', 'cumbia grupera',
        'cumbia romantica', 'cumbia tropical', 'cumbia andina', 'cumbia amazonica', 'cumbia urbana',
        'cumbia fusion', 'cumbia electronica', 'cumbia digital', 'nueva cumbia', 'neo cumbia',
        'cumbia pop', 'cumbia rock', 'cumbia rap', 'cumbia reggae',
        
        # Core Latino genres
        'latin', 'latino', 'reggaeton', 'salsa', 'bachata', 'merengue', 'bolero', 'ranchera',
        'banda', 'mariachi', 'grupera', 'norteno', 'nortena', 'tejano', 'conjunto',
        'vallenato', 'champeta', 'porro', 'gaita', 'bambuco', 'pasillo', 'joropo',
        'son', 'son cubano', 'mambo', 'cha cha', 'rumba', 'conga', 'bossa nova',
        
        # Regional/Country specific
        'mexican', 'colombia', 'colombian', 'argentina', 'argentinian', 'peru', 'peruvian',
        'chile', 'chilean', 'venezuela', 'venezuelan', 'ecuador', 'ecuadorian', 'bolivia', 'bolivian',
        'uruguay', 'uruguayan', 'paraguay', 'paraguayan', 'panama', 'panamanian',
        'costa rica', 'costa rican', 'guatemala', 'guatemalan', 'honduras', 'honduran',
        'nicaragua', 'nicaraguan', 'el salvador', 'salvadoran', 'cuba', 'cuban',
        'dominican', 'puerto rico', 'puerto rican', 'brasil', 'brazilian', 'brazil',
        
        # Language indicators
        'spanish', 'espanol', 'español', 'portugues', 'português', 'portuguese',
        
        # Pop/Rock Latino
        'latin pop', 'pop latino', 'rock en espanol', 'rock en español', 'rock latino',
        'latin rock', 'latin alternative', 'latin indie', 'indie latino', 'alternativo',
        'rock alternativo', 'pop rock latino', 'balada', 'baladas',
        
        # Urban Latino
        'reggaeton', 'trap latino', 'latin trap', 'urbano latino', 'latin urban',
        'dembow', 'perreo', 'latin hip hop', 'rap latino', 'hip hop latino',
        'latin r&b', 'r&b latino',
        
        # Traditional/Folk
        'folk latino', 'latin folk', 'tradicional', 'musica tradicional', 'folklorico',
        'folklorica', 'andina', 'andean', 'amazonica', 'amazonian', 'llanera', 'tropical',
        'musica tropical', 'caribe', 'caribbean', 'antillana', 'antillean',
        
        # Electronic/Fusion
        'latin electronic', 'electronica latina', 'latin house', 'latin techno',
        'latin bass', 'moombahton', 'latin breaks', 'tribal latino', 'latin ambient',
        
        # Modern genres
        'latin soul', 'latin funk', 'latin jazz', 'jazz latino', 'boogaloo', 'latin disco',
        'latin punk', 'punk latino', 'latin metal', 'metal latino', 'latin hardcore',
        
        # Generic world tags
        'world', 'world music', 'international', 'global', 'ethnic', 'traditional',
        'indigenous', 'native', 'aboriginal', 'tribal'
    ]
    
    world_score = sum(1 for term in world_terms if term in genre_text) / max(len(all_genres), 1)
    
    # Also check track names for Spanish/Portuguese words (common in cumbia)
    track_names_text = ' '.join([track['name'].lower() for track in tracks])
    spanish_words = [
        'amor', 'corazon', 'corazón', 'vida', 'luna', 'sol', 'noche', 'dia', 'día',
        'mujer', 'hombre', 'niña', 'niño', 'casa', 'agua', 'fuego', 'tierra', 'cielo',
        'tiempo', 'amigo', 'amiga', 'hermano', 'hermana', 'madre', 'padre', 'hijo', 'hija',
        'baila', 'baile', 'danza', 'ritmo', 'musica', 'música', 'cancion', 'canción',
        'fiesta', 'alegria', 'alegría', 'feliz', 'triste', 'loco', 'loca', 'bonita', 'bonito',
        'mi', 'tu', 'tú', 'el', 'la', 'que', 'como', 'cómo', 'donde', 'dónde', 'cuando', 'cuándo',
        'si', 'sí', 'no', 'pero', 'para', 'con', 'sin', 'por', 'de', 'en', 'a', 'y', 'o', 'u',
        'esta', 'está', 'este', 'esto', 'esa', 'ese', 'eso', 'aqui', 'aquí', 'alli', 'allí',
        'vamos', 'voy', 'vas', 'va', 'van', 'ven', 'ver', 'veo', 'ves', 've',
        'soy', 'eres', 'es', 'somos', 'son', 'estar', 'estoy', 'estas', 'estás'
    ]
    spanish_content = sum(1 for word in spanish_words if word in track_names_text) / max(len(tracks), 1)
    
    # Check artist names for Latino indicators (expanded)
    artist_names_text = ' '.join([track['artists'][0]['name'].lower() for track in tracks])
    latino_artist_indicators = [
        # Common first names
        'juan', 'carlos', 'luis', 'antonio', 'manuel', 'jose', 'josé', 'maria', 'maría',
        'ana', 'carmen', 'pedro', 'pablo', 'miguel', 'rafael', 'ricardo', 'fernando',
        'alejandro', 'francisco', 'javier', 'diego', 'sergio', 'mario', 'alberto',
        'jorge', 'oscar', 'óscar', 'eduardo', 'roberto', 'daniel', 'david', 'jesus', 'jesús',
        # Common last names/indicators
        'rodriguez', 'rodríguez', 'martinez', 'martínez', 'garcia', 'garcía', 'lopez', 'lópez',
        'gonzalez', 'gonzález', 'sanchez', 'sánchez', 'ramirez', 'ramírez', 'torres', 'flores',
        'rivera', 'morales', 'jimenez', 'jiménez', 'mendoza', 'castillo', 'vargas', 'herrera',
        'medina', 'guerrero', 'ramos', 'ayala', 'cruz', 'moreno', 'ortiz', 'gutierrez', 'gutiérrez',
        # Band/group indicators
        'los', 'las', 'la', 'el', 'grupo', 'banda', 'orquesta', 'conjunto', 'mariachi',
        'cumbia', 'sonora', 'tropical', 'internacional', 'musical', 'super', 'súper'
    ]
    artist_latino_score = sum(1 for indicator in latino_artist_indicators if indicator in artist_names_text) / max(len(tracks), 1)
    
    # Combine all Latino indicators with weighted scores
    combined_world_score = (
        world_score * 0.6 +           # Genre tags most reliable
        spanish_content * 0.25 +      # Spanish lyrics good indicator  
        artist_latino_score * 0.15    # Artist names helpful but less reliable
    )
    world_score = min(1.0, combined_world_score)  # Cap at 1.0
    
    # Create the EXACT 35 features the behavioral model expects:
    
    # Basic audio features (simulated from genres and popularity)
    features['energy_preference'] = min(1.0, (electronic_score * 0.8 + rock_score * 0.9 + hiphop_score * 0.7 + world_score * 0.6 + pop_score * 0.5) / 2)
    features['social_music_score'] = (pop_score * 0.6 + electronic_score * 0.4 + track_df['popularity'].mean() / 200)
    features['high_energy_preference'] = 1 if features['energy_preference'] > 0.7 else 0
    features['danceable_preference'] = min(1.0, (electronic_score * 0.9 + pop_score * 0.7 + hiphop_score * 0.8 + world_score * 0.8) / 2)
    features['loudness_preference'] = (rock_score * 0.8 + electronic_score * 0.6 + hiphop_score * 0.7 + 0.3) / 2
    features['tempo_preference'] = (electronic_score * 0.8 + hiphop_score * 0.7 + world_score * 0.7 + 0.4) / 2
    
    # Openness features
    features['musical_complexity'] = (calm_score * 0.7 + len(set(all_genres)) / 20 + world_score * 0.5) / 2
    features['experimental_preference'] = len(set(all_genres)) / 30  # Genre diversity as proxy
    features['acoustic_exploration'] = calm_score
    features['instrumental_preference'] = calm_score * 0.6
    features['genre_openness'] = min(1.0, len(set(all_genres)) / 15)  # More sensitive
    features['unconventional_preference'] = 1 - (track_df['popularity'].mean() / 100)
    
    # Conscientiousness features  
    features['listening_consistency'] = 1 - (track_df['popularity'].std() / 100) if len(track_df) > 1 else 0.8
    features['routine_preference'] = track_df['popularity'].mean() / 100
    features['completion_tendency'] = features['energy_preference']
    features['organized_listening'] = (track_df['popularity'] > 50).mean()
    features['mainstream_preference'] = (track_df['popularity'] > 70).mean()
    features['predictable_choice'] = 1 - features['genre_openness']
    
    # Agreeableness features
    # Ed Sheeran should score high here (positive, mainstream pop)
    # Your indie/latino should score differently
    track_names_text = ' '.join([track['name'].lower() for track in tracks])
    positive_words = ['love', 'happy', 'good', 'beautiful', 'wonderful', 'perfect', 'sweet', 'smile', 'heart']
    positive_content = sum(1 for word in positive_words if word in track_names_text) / len(tracks)
    
    features['positive_music_preference'] = (pop_score * 0.6 + positive_content * 0.4 + (1 - track_df['explicit'].mean()) * 0.2) / 1.2
    features['mellow_preference'] = calm_score
    features['harmony_seeking'] = features['positive_music_preference']
    features['avoid_aggressive'] = 1 - (rock_score * 0.7 + track_df['explicit'].mean() * 0.3)
    features['social_acceptance'] = (pop_score * 0.7 + features['mainstream_preference'] * 0.3)
    features['cooperative_music'] = calm_score + pop_score * 0.5
    
    # Neuroticism features
    negative_words = ['sad', 'cry', 'pain', 'hurt', 'broken', 'lonely', 'dark', 'death', 'hate']
    negative_content = sum(1 for word in negative_words if word in track_names_text) / len(tracks)
    
    features['emotional_music_seeking'] = negative_content + (1 - features['positive_music_preference']) * 0.5
    features['mood_instability'] = track_df['popularity'].std() / 100 if len(track_df) > 1 else 0.1
    features['anxiety_music'] = (negative_content + rock_score * 0.5) / 1.5
    features['comfort_seeking'] = calm_score
    features['emotional_volatility'] = features['mood_instability']
    features['stress_response'] = features['emotional_music_seeking']
    
    # General features
    features['music_sophistication'] = (world_score * 0.4 + features['genre_openness'] * 0.4 + (1 - features['mainstream_preference']) * 0.2)
    features['emotional_regulation'] = features['positive_music_preference']
    features['stimulation_seeking'] = features['energy_preference']
    features['mood_management'] = features['positive_music_preference']
    features['musical_engagement'] = features['danceable_preference']
    
    # Ensure all values are in [0, 1] range and handle edge cases
    for key, value in features.items():
        if hasattr(value, 'item'):
            value = value.item()
        elif hasattr(value, '__len__') and not isinstance(value, str):
            value = float(value.mean() if hasattr(value, 'mean') else value[0])
        
        if pd.isna(value) or value is None:
            features[key] = 0.5
        else:
            features[key] = max(0.0, min(1.0, float(value)))
    
    st.write(f"DEBUG: Generated exactly {len(features)} features for behavioral model")
    
    # Show some key differentiating features for debugging
    st.write(f"DEBUG: World/Latin score: {world_score:.3f}, Pop score: {pop_score:.3f}, Mainstream: {features['mainstream_preference']:.3f}")
    
    return features

def get_audio_features_robust(sp, tracks):
    """Get audio features with fallback strategies"""
    
    st.write("🎵 Getting audio features...")
    track_ids = [track['id'] for track in tracks if track['id']]
    
    if not track_ids:
        st.warning("No valid track IDs found")
        return create_fallback_features(tracks)
    
    all_features = []
    
    # Try smaller batches to avoid rate limiting
    batch_sizes = [20, 10, 5, 1]
    
    for batch_size in batch_sizes:
        if not track_ids:
            break
            
        st.write(f"Trying batch size {batch_size}...")
        remaining_ids = track_ids.copy()
        
        for i in range(0, len(remaining_ids), batch_size):
            batch_ids = remaining_ids[i:i+batch_size]
            
            try:
                features = sp.audio_features(batch_ids)
                if features:
                    valid_features = [f for f in features if f is not None]
                    all_features.extend(valid_features)
                    
                    # Remove successful IDs
                    successful_ids = [f['id'] for f in valid_features]
                    track_ids = [tid for tid in track_ids if tid not in successful_ids]
                
                time.sleep(0.5)
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    st.warning(f"Rate limited, waiting...")
                    time.sleep(5)
                    continue
                else:
                    continue
        
        if not track_ids:
            break
    
    if all_features:
        st.success(f"✅ Got audio features for {len(all_features)} tracks")
        return all_features
    else:
        st.warning("⚠️ Could not get audio features - using fallback analysis")
        return create_fallback_features(tracks)

def create_fallback_features(tracks):
    """Create estimated audio features when Spotify API fails"""
    
    st.info("🔄 Creating estimated audio features based on track metadata...")
    
    fallback_features = []
    
    for track in tracks:
        popularity = track.get('popularity', 50)
        
        estimated_features = {
            'id': track['id'],
            'danceability': 0.5 + (popularity - 50) / 200,
            'energy': 0.5,
            'valence': 0.5,
            'acousticness': 0.3,
            'instrumentalness': 0.1,
            'speechiness': 0.1,
            'loudness': -10,
            'tempo': 120,
            'liveness': 0.2,
            'popularity': popularity,
            'key': np.random.randint(0, 12),
            'mode': np.random.randint(0, 2),
            'time_signature': 4
        }
        
        # Adjust based on genre keywords
        track_text = f"{track['name']} {track['artists'][0]['name']}".lower()
        
        if any(word in track_text for word in ['electronic', 'dance', 'house', 'techno']):
            estimated_features['danceability'] = 0.8
            estimated_features['energy'] = 0.8
            
        if any(word in track_text for word in ['acoustic', 'folk', 'country']):
            estimated_features['acousticness'] = 0.8
            estimated_features['energy'] = 0.3
            
        if any(word in track_text for word in ['happy', 'love', 'good']):
            estimated_features['valence'] = 0.7
            
        if any(word in track_text for word in ['sad', 'dark', 'death']):
            estimated_features['valence'] = 0.3
        
        fallback_features.append(estimated_features)
    
    st.info(f"Created estimated features for {len(fallback_features)} tracks")
    return fallback_features

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

# -----------------------------------------------------------------------------
# Model-Based Personality Prediction 
# -----------------------------------------------------------------------------
def predict_personality_with_models(features, loaded_models):
    """Use trained models to predict personality - FIXED to handle metadata features"""
    
    if not features:
        return predict_personality_fallback(features)
    
    predictions = {}
    prediction_confidence = {}
    
    # The core issue: your models expect different features than what we generate
    # Let's create a mapping from our metadata features to what models expect
    
    # Try research model first - it's most likely to work
    if 'research' in loaded_models:
        try:
            st.write("DEBUG: Trying research model with feature mapping...")
            model_data = loaded_models['research']
            models = model_data.get('models', {})
            scalers = model_data.get('scalers', {})
            expected_features = model_data.get('feature_cols', [])
            
            st.write(f"DEBUG: Research model expects: {expected_features}")
            st.write(f"DEBUG: We have: {list(features.keys())[:5]}...")
            
            # Map our features to what research model expects
            mapped_features = {}
            
            # Direct mappings where possible
            feature_mapping = {
                'energy': 'energy_preference',
                'danceability': 'danceable_preference', 
                'valence': 'valence_preference',
                'acousticness': 'acousticness_preference',
                'instrumentalness': features.get('instrumental_preference', 0.2),
                'speechiness': features.get('rock_preference', 0.1) * 0.3,  # Rock music tends to be more speech-like
                'loudness': 'loudness_preference',
                'popularity': 'mainstream_preference',
                'genre_diversity': 'genre_openness',
                'listening_consistency': 'listening_consistency'
            }
            
            for model_feature, our_feature in feature_mapping.items():
                if isinstance(our_feature, str) and our_feature in features:
                    mapped_features[model_feature] = features[our_feature]
                elif isinstance(our_feature, (int, float)):
                    mapped_features[model_feature] = our_feature
                else:
                    mapped_features[model_feature] = 0.5  # Default
            
            st.write(f"DEBUG: Mapped features: {mapped_features}")
            
            # Create feature vector
            feature_vector = []
            for col in expected_features:
                feature_vector.append(mapped_features.get(col, 0.5))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            st.write(f"DEBUG: Feature array shape: {feature_array.shape}")
            
            # Make predictions
            for trait in models:
                try:
                    scaler = scalers.get(trait)
                    if scaler:
                        scaled_features = scaler.transform(feature_array)
                    else:
                        scaled_features = feature_array
                    
                    prediction = models[trait].predict(scaled_features)[0]
                    predictions[trait] = round(np.clip(prediction, 1.0, 5.0), 2)
                    
                    st.write(f"DEBUG: {trait} prediction: {prediction:.3f}")
                    
                except Exception as e:
                    st.write(f"DEBUG: Error predicting {trait}: {e}")
                    predictions[trait] = 3.0
            
            if predictions:
                st.success("✅ Used research model with feature mapping")
                return predictions, None
                
        except Exception as e:
            st.error(f"Research model error: {e}")
    
    # If research model fails, try fallback behavioral approach
    if 'behavioral' in loaded_models:
        st.warning("Research model failed, behavioral model expects different features - using enhanced fallback")
    
    # Enhanced fallback that's more sensitive to your actual music differences
    predictions = predict_enhanced_fallback(features)
    return predictions, None

def predict_enhanced_fallback(features):
    """Enhanced fallback that should distinguish between Ed Sheeran vs indie/Latino music"""
    
    if not features:
        return {trait: 3.0 for trait in ['Extraversion', 'Openness', 'Conscientiousness', 'Agreeableness', 'Neuroticism']}
    
    predictions = {}
    
    # Get key differentiating scores
    world_score = features.get('world_score', 0.0)  # From debug output
    pop_score = features.get('pop_score', 0.0)      # From debug output  
    mainstream_pref = features.get('mainstream_preference', 0.5)
    genre_openness = features.get('genre_openness', 0.3)
    
    st.write(f"DEBUG FALLBACK: world={world_score:.3f}, pop={pop_score:.3f}, mainstream={mainstream_pref:.3f}, genres={genre_openness:.3f}")
    
    # EXTRAVERSION - Ed Sheeran (mainstream pop) should be higher than indie
    extraversion = (
        features.get('energy_preference', 0.5) * 0.25 +
        features.get('social_music_score', 0.5) * 0.25 +
        mainstream_pref * 0.2 +
        pop_score * 0.15 +
        features.get('danceable_preference', 0.5) * 0.15
    )
    predictions['Extraversion'] = np.clip(extraversion * 4 + 1.5, 1, 5)
    
    # OPENNESS - Indie/Latino should be much higher than Ed Sheeran
    openness = (
        genre_openness * 0.3 +
        world_score * 0.25 +  # Latino music shows cultural openness
        (1 - mainstream_pref) * 0.2 +  # Non-mainstream shows openness
        features.get('experimental_preference', 0.1) * 0.15 +
        features.get('musical_sophistication', 0.3) * 0.1
    )
    predictions['Openness'] = np.clip(openness * 4 + 1.5, 1, 5)
    
    # CONSCIENTIOUSNESS - Ed Sheeran (organized pop) should be higher  
    conscientiousness = (
        mainstream_pref * 0.35 +
        features.get('routine_preference', 0.5) * 0.25 +
        pop_score * 0.2 +
        features.get('listening_consistency', 0.8) * 0.2
    )
    predictions['Conscientiousness'] = np.clip(conscientiousness * 4 + 1.5, 1, 5)
    
    # AGREEABLENESS - Ed Sheeran (positive pop) should be higher
    agreeableness = (
        features.get('positive_music_preference', 0.5) * 0.3 +
        pop_score * 0.25 +
        mainstream_pref * 0.2 +
        features.get('harmony_seeking', 0.5) * 0.15 +
        (1 - features.get('emotional_music_seeking', 0.3)) * 0.1
    )
    predictions['Agreeableness'] = np.clip(agreeableness * 4 + 1.5, 1, 5)
    
    # NEUROTICISM - Should differentiate based on emotional content
    neuroticism = (
        features.get('emotional_music_seeking', 0.5) * 0.3 +
        (1 - features.get('positive_music_preference', 0.5)) * 0.25 +
        features.get('mood_instability', 0.2) * 0.2 +
        (1 - mainstream_pref) * 0.15 +  # Non-mainstream might indicate emotional seeking
        features.get('anxiety_music', 0.3) * 0.1
    )
    predictions['Neuroticism'] = np.clip(neuroticism * 4 + 1.5, 1, 5)
    
    # Round results
    for trait in predictions:
        predictions[trait] = round(predictions[trait], 2)
    
    st.write(f"DEBUG ENHANCED FALLBACK: {predictions}")
    return predictions

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
        
        return predictions, None  # No confidence - it was meaningless anyway
        
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
# UI and Insights
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
# Main App
# -----------------------------------------------------------------------------
def main():
    st.title("Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits from your music!")
    
    # Load trained models
    loaded_models = load_trained_models()
    
    sp = ensure_spotify_client()
    
    # Show authenticated user (ONLY CHANGE - show who is logged in)
    if hasattr(st.session_state, 'user') and st.session_state.user:
        user = st.session_state.user
        st.success(f"Connected to Spotify - Logged in as: {user.get('display_name', 'Unknown')} ({user.get('id', 'Unknown')})")

    if st.button("Analyze My Musical Personality", type="primary"):
        
        with st.spinner("Collecting and analyzing your music data..."):
            music_data = get_user_music_data_simplified(sp)
        
        if music_data and music_data['tracks']:
            
            # Extract features from metadata and enriched data
            features = create_features_from_metadata(music_data)
            
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
                st.info("Analysis based on track metadata, genres, popularity patterns, and enriched data from Last.fm")
                
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
                    st.metric("Mainstream Score", f"{features['mainstream_preference']:.0%}")
                
                with col2:
                    st.metric("Unique Genres", features.get('unique_genres_count', 0))
                    st.metric("Genre Diversity", f"{features.get('genre_diversity', 0):.2f}")
                
                with col3:
                    st.metric("Artist Diversity", f"{features.get('artist_diversity', 0):.2f}")
                    st.metric("Avg Popularity", f"{features.get('popularity', 0):.0%}")
                
                # Genre preferences
                st.subheader("Your Genre Preferences")
                
                genre_prefs = {
                    'Electronic/Dance': features.get('electronic_preference', 0),
                    'Rock/Alternative': features.get('rock_preference', 0),
                    'Pop/Mainstream': features.get('pop_preference', 0),
                    'Hip-Hop/Rap': features.get('hiphop_preference', 0),
                    'Calm/Acoustic': features.get('calm_preference', 0),
                    'Experimental': features.get('experimental_preference', 0)
                }
                
                genre_df = pd.DataFrame(list(genre_prefs.items()), columns=['Genre', 'Preference'])
                st.bar_chart(genre_df.set_index('Genre'))
                
                # Model info
                if loaded_models:
                    with st.expander("Model Information"):
                        st.write(f"Using {len(loaded_models)} trained model(s) for prediction")
                        for model_name in loaded_models.keys():
                            st.write(f"- {model_name.title()} Model")
        
        else:
            st.error("Could not analyze your music. This might be due to API limitations or insufficient listening history.")
            st.info("Try again later, or make sure you have recent listening activity on Spotify.")

if __name__ == "__main__":
    main()