# app.py - Clean Spotify Personality Predictor with Proper Authentication
import os
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib
import uuid
from datetime import datetime
import random

# Configuration
SCOPES = "user-top-read user-library-read user-read-recently-played user-follow-read playlist-read-private"

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET") 
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

# Validate credentials
if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
    st.error("Missing Spotify API credentials. Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI environment variables.")
    st.stop()

def clear_authentication_state():
    """Clear all authentication-related session state"""
    keys_to_clear = [k for k in st.session_state.keys() if 'token' in k.lower() or 'auth' in k.lower()]
    for key in keys_to_clear:
        del st.session_state[key]

def get_unique_session_id():
    """Generate or retrieve unique session identifier"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def create_spotify_auth_manager():
    """Create Spotify OAuth manager with unique state"""
    session_id = get_unique_session_id()
    unique_state = f"spotify_auth_{session_id}_{int(time.time())}"
    
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=None,
        state=unique_state,
        show_dialog=True  # Force login dialog
    )

def authenticate_spotify():
    """Handle Spotify authentication flow"""
    auth_manager = create_spotify_auth_manager()
    
    # Check for authorization code in URL
    params = st.query_params
    
    if "code" in params:
        code = params["code"]
        received_state = params.get("state", "")
        
        # Verify state parameter
        session_id = get_unique_session_id()
        if not received_state.startswith(f"spotify_auth_{session_id}"):
            st.error("Authentication state mismatch. Please try logging in again.")
            st.query_params.clear()
            st.rerun()
        
        try:
            # Exchange code for token
            token_info = auth_manager.get_access_token(code, as_dict=True)
            
            if token_info and 'access_token' in token_info:
                # Store token with session-specific key
                token_key = f"spotify_token_{session_id}"
                st.session_state[token_key] = token_info
                
                # Clear URL parameters
                st.query_params.clear()
                st.success("Successfully authenticated with Spotify!")
                st.rerun()
            else:
                st.error("Failed to obtain access token")
                return None
                
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return None
    
    # Check for existing valid token
    session_id = get_unique_session_id()
    token_key = f"spotify_token_{session_id}"
    token_info = st.session_state.get(token_key)
    
    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])
    
    # Need fresh authentication
    auth_url = auth_manager.get_authorize_url()
    
    st.warning("Please authenticate with your Spotify account")
    st.markdown(f"[Login with Spotify]({auth_url})")
    
    with st.expander("Authentication Debug Info"):
        st.write(f"Session ID: {get_unique_session_id()}")
        st.write(f"Expected state prefix: spotify_auth_{get_unique_session_id()}")
        
        if st.button("Clear Authentication State"):
            clear_authentication_state()
            st.rerun()
    
    return None

def verify_user_identity(sp):
    """Verify and display current user identity"""
    try:
        user_profile = sp.current_user()
        
        user_info = {
            'id': user_profile.get('id', 'unknown'),
            'display_name': user_profile.get('display_name', 'Unknown'),
            'email': user_profile.get('email', 'Not provided'),
            'country': user_profile.get('country', 'Unknown'),
            'followers': user_profile.get('followers', {}).get('total', 0),
            'product': user_profile.get('product', 'Unknown'),
            'external_urls': user_profile.get('external_urls', {}).get('spotify', '')
        }
        
        st.success(f"Authenticated as: {user_info['display_name']} ({user_info['id']})")
        
        return user_info
        
    except Exception as e:
        st.error(f"Failed to verify user identity: {e}")
        return None

def collect_spotify_data(sp, user_info, limit=50):
    """Collect comprehensive Spotify data for analysis"""
    
    st.write("Collecting your Spotify data...")
    
    data = {
        'user_info': user_info,
        'collection_time': datetime.now().isoformat(),
        'session_id': get_unique_session_id(),
        'tracks': [],
        'artists': [],
        'audio_features': [],
        'genres': [],
        'playlists': [],
        'following': [],
        'recent_tracks': []
    }
    
    # Collect data with error handling
    try:
        # Top tracks across different time ranges
        all_tracks = []
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                response = sp.current_user_top_tracks(limit=limit//3, time_range=time_range)
                tracks = response.get('items', [])
                all_tracks.extend(tracks)
                st.write(f"Collected {len(tracks)} {time_range} top tracks")
            except Exception as e:
                st.warning(f"Could not get {time_range} tracks: {e}")
        
        # Recently played tracks
        try:
            response = sp.current_user_recently_played(limit=limit)
            recent_items = response.get('items', [])
            recent_tracks = [item['track'] for item in recent_items if item.get('track')]
            all_tracks.extend(recent_tracks)
            data['recent_tracks'] = recent_tracks
            st.write(f"Collected {len(recent_tracks)} recently played tracks")
        except Exception as e:
            st.warning(f"Could not get recent tracks: {e}")
        
        # Remove duplicates
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track and track.get('id') and track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        data['tracks'] = unique_tracks
        st.success(f"Found {len(unique_tracks)} unique tracks")
        
        # Get audio features for tracks
        if unique_tracks:
            try:
                track_ids = [track['id'] for track in unique_tracks if track.get('id')]
                
                # Process in batches to avoid rate limits
                audio_features = []
                batch_size = 50
                
                for i in range(0, len(track_ids), batch_size):
                    batch_ids = track_ids[i:i+batch_size]
                    try:
                        features = sp.audio_features(batch_ids)
                        audio_features.extend([f for f in features if f is not None])
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        st.warning(f"Could not get audio features for batch {i//batch_size + 1}: {e}")
                
                data['audio_features'] = audio_features
                st.write(f"Collected audio features for {len(audio_features)} tracks")
                
            except Exception as e:
                st.warning(f"Could not get audio features: {e}")
        
        # Get artist information and genres
        if unique_tracks:
            try:
                artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks if track.get('artists')]))
                
                artists = []
                genres = []
                batch_size = 50
                
                for i in range(0, len(artist_ids), batch_size):
                    batch_ids = artist_ids[i:i+batch_size]
                    try:
                        artist_response = sp.artists(batch_ids)
                        batch_artists = artist_response.get('artists', [])
                        artists.extend(batch_artists)
                        
                        for artist in batch_artists:
                            genres.extend(artist.get('genres', []))
                        
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        st.warning(f"Could not get artists for batch {i//batch_size + 1}: {e}")
                
                data['artists'] = artists
                data['genres'] = genres
                st.write(f"Collected {len(artists)} artists and {len(genres)} genre tags")
                
            except Exception as e:
                st.warning(f"Could not get artist information: {e}")
        
        # Get top artists
        try:
            response = sp.current_user_top_artists(limit=20)
            top_artists = response.get('items', [])
            data['top_artists'] = top_artists
            st.write(f"Collected {len(top_artists)} top artists")
        except Exception as e:
            st.warning(f"Could not get top artists: {e}")
        
        # Get playlists
        try:
            response = sp.current_user_playlists(limit=20)
            playlists = response.get('items', [])
            data['playlists'] = playlists
            st.write(f"Collected {len(playlists)} playlists")
        except Exception as e:
            st.warning(f"Could not get playlists: {e}")
        
        # Create data fingerprint
        fingerprint_data = {
            'user_id': user_info['id'],
            'track_count': len(data['tracks']),
            'genre_count': len(data['genres']),
            'artist_count': len(data['artists']),
            'track_sample': [t['name'] for t in data['tracks'][:10]],
            'genre_sample': list(set(data['genres']))[:10]
        }
        
        fingerprint_str = str(fingerprint_data)
        data['fingerprint'] = hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
        
        st.success(f"Data collection complete! Fingerprint: {data['fingerprint']}")
        
        return data
        
    except Exception as e:
        st.error(f"Error during data collection: {e}")
        return None

def extract_music_features(data):
    """Extract features for personality prediction"""
    
    if not data or not data['tracks']:
        return None
    
    tracks = data['tracks']
    audio_features = data['audio_features']
    genres = data['genres']
    
    # Basic track statistics
    popularities = [track.get('popularity', 50) for track in tracks]
    durations = [track.get('duration_ms', 200000) for track in tracks]
    explicit_count = sum(1 for track in tracks if track.get('explicit', False))
    
    features = {
        # Basic preferences
        'avg_popularity': np.mean(popularities) / 100,
        'mainstream_preference': sum(1 for p in popularities if p > 70) / len(popularities),
        'underground_preference': sum(1 for p in popularities if p < 30) / len(popularities),
        'explicit_content_ratio': explicit_count / len(tracks),
        'avg_track_length': np.mean(durations) / 300000,  # Normalized to ~5 minutes
        
        # Diversity metrics
        'artist_diversity': len(set(track['artists'][0]['name'] for track in tracks)) / len(tracks),
        'genre_diversity': len(set(genres)) / max(len(genres), 1) if genres else 0,
        'unique_genres_count': len(set(genres)),
        
        # Audio feature analysis
        'energy': 0.5,
        'danceability': 0.5,
        'valence': 0.5,
        'acousticness': 0.5,
        'instrumentalness': 0.5,
        'speechiness': 0.5,
        'loudness': 0.5,
        'tempo': 0.5
    }
    
    # Extract audio features if available
    if audio_features:
        audio_df = pd.DataFrame(audio_features)
        
        for feature in ['energy', 'danceability', 'valence', 'acousticness', 
                       'instrumentalness', 'speechiness']:
            if feature in audio_df.columns:
                features[feature] = audio_df[feature].mean()
        
        if 'loudness' in audio_df.columns:
            # Normalize loudness (typically -60 to 0 dB)
            features['loudness'] = (audio_df['loudness'].mean() + 60) / 60
        
        if 'tempo' in audio_df.columns:
            # Normalize tempo (typically 50-200 BPM)
            features['tempo'] = np.clip(audio_df['tempo'].mean() / 200, 0, 1)
    
    # Genre analysis
    if genres:
        genre_text = ' '.join(genres).lower()
        
        # Define genre categories
        electronic_terms = ['electronic', 'house', 'techno', 'edm', 'dance', 'trance']
        rock_terms = ['rock', 'metal', 'punk', 'alternative', 'grunge', 'indie rock']
        pop_terms = ['pop', 'mainstream', 'chart', 'radio']
        classical_terms = ['classical', 'orchestral', 'symphony', 'opera']
        jazz_terms = ['jazz', 'blues', 'swing', 'bebop']
        folk_terms = ['folk', 'country', 'americana', 'bluegrass']
        hiphop_terms = ['hip hop', 'rap', 'hip-hop', 'trap']
        
        features['electronic_preference'] = sum(1 for term in electronic_terms if term in genre_text) / max(len(genres), 1)
        features['rock_preference'] = sum(1 for term in rock_terms if term in genre_text) / max(len(genres), 1)
        features['pop_preference'] = sum(1 for term in pop_terms if term in genre_text) / max(len(genres), 1)
        features['classical_preference'] = sum(1 for term in classical_terms if term in genre_text) / max(len(genres), 1)
        features['jazz_preference'] = sum(1 for term in jazz_terms if term in genre_text) / max(len(genres), 1)
        features['folk_preference'] = sum(1 for term in folk_terms if term in genre_text) / max(len(genres), 1)
        features['hiphop_preference'] = sum(1 for term in hiphop_terms if term in genre_text) / max(len(genres), 1)
    else:
        # Default values if no genres available
        for pref in ['electronic_preference', 'rock_preference', 'pop_preference', 
                    'classical_preference', 'jazz_preference', 'folk_preference', 'hiphop_preference']:
            features[pref] = 0.0
    
    # Ensure all values are in [0, 1] range
    for key, value in features.items():
        if pd.isna(value) or value is None:
            features[key] = 0.5
        else:
            features[key] = np.clip(float(value), 0.0, 1.0)
    
    return features

def predict_personality(features):
    """Predict Big Five personality traits from music features"""
    
    if not features:
        return None
    
    predictions = {}
    
    # OPENNESS TO EXPERIENCE
    # High openness: diverse genres, experimental music, less mainstream
    openness = (
        features['genre_diversity'] * 0.25 +
        features['artist_diversity'] * 0.2 +
        (1 - features['mainstream_preference']) * 0.2 +
        features['classical_preference'] * 0.1 +
        features['jazz_preference'] * 0.1 +
        features['instrumentalness'] * 0.1 +
        (1 - features['pop_preference']) * 0.05
    )
    predictions['Openness'] = np.clip(openness * 4 + 1, 1, 5)
    
    # CONSCIENTIOUSNESS
    # High conscientiousness: mainstream music, consistent preferences, less explicit
    conscientiousness = (
        features['mainstream_preference'] * 0.3 +
        (1 - features['explicit_content_ratio']) * 0.25 +
        features['pop_preference'] * 0.2 +
        (1 - features['genre_diversity']) * 0.15 +
        features['valence'] * 0.1
    )
    predictions['Conscientiousness'] = np.clip(conscientiousness * 4 + 1, 1, 5)
    
    # EXTRAVERSION
    # High extraversion: energetic, danceable, popular music
    extraversion = (
        features['energy'] * 0.25 +
        features['danceability'] * 0.25 +
        features['electronic_preference'] * 0.15 +
        features['pop_preference'] * 0.15 +
        features['mainstream_preference'] * 0.1 +
        features['valence'] * 0.1
    )
    predictions['Extraversion'] = np.clip(extraversion * 4 + 1, 1, 5)
    
    # AGREEABLENESS
    # High agreeableness: positive, harmonious music, less aggressive
    agreeableness = (
        features['valence'] * 0.3 +
        features['acousticness'] * 0.2 +
        (1 - features['explicit_content_ratio']) * 0.2 +
        features['folk_preference'] * 0.1 +
        features['pop_preference'] * 0.1 +
        (1 - features['energy']) * 0.1
    )
    predictions['Agreeableness'] = np.clip(agreeableness * 4 + 1, 1, 5)
    
    # NEUROTICISM
    # High neuroticism: emotional music, less positive valence
    neuroticism = (
        (1 - features['valence']) * 0.3 +
        features['explicit_content_ratio'] * 0.2 +
        features['rock_preference'] * 0.15 +
        features['energy'] * 0.15 +
        (1 - features['mainstream_preference']) * 0.1 +
        features['speechiness'] * 0.1
    )
    predictions['Neuroticism'] = np.clip(neuroticism * 4 + 1, 1, 5)
    
    # Round to reasonable precision
    for trait in predictions:
        predictions[trait] = round(predictions[trait], 2)
    
    return predictions

def display_results(data, features, predictions):
    """Display analysis results"""
    
    user_info = data['user_info']
    
    st.header("Your Musical Personality Analysis")
    
    # User verification
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("User Verification")
        st.write(f"**Name:** {user_info['display_name']}")
        st.write(f"**User ID:** {user_info['id']}")
        st.write(f"**Country:** {user_info['country']}")
        st.write(f"**Data Fingerprint:** `{data['fingerprint']}`")
    
    with col2:
        st.metric("Tracks Analyzed", len(data['tracks']))
        st.metric("Unique Genres", features['unique_genres_count'])
        st.metric("Artist Diversity", f"{features['artist_diversity']:.2f}")
    
    # Personality radar chart
    if predictions:
        st.subheader("Big Five Personality Traits")
        
        fig = go.Figure()
        
        traits = list(predictions.keys())
        scores = list(predictions.values())
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=traits,
            fill='toself',
            name='Your Personality',
            line_color='rgb(46, 204, 113)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, 5],
                    tickvals=[1, 2, 3, 4, 5]
                )
            ),
            showlegend=False,
            title="Your Musical Personality Profile",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed trait descriptions
        trait_descriptions = {
            'Openness': {
                'high': "You love exploring new musical territories and unconventional sounds!",
                'medium': "You balance familiar favorites with occasional musical exploration.",
                'low': "You prefer sticking to tried-and-true musical styles you know you'll enjoy."
            },
            'Conscientiousness': {
                'high': "Your music choices tend to be thoughtful and organized.",
                'medium': "You show some structure in your musical preferences while staying flexible.",
                'low': "Your musical taste is spontaneous and driven by mood and impulse!"
            },
            'Extraversion': {
                'high': "You gravitate toward energetic, social music that gets you moving!",
                'medium': "You enjoy both upbeat social music and quieter personal listening.",
                'low': "You prefer introspective, calmer music for personal reflection."
            },
            'Agreeableness': {
                'high': "You love harmonious, positive music that brings people together!",
                'medium': "You appreciate both uplifting and more complex emotional music.",
                'low': "You're drawn to more intense, unconventional, or challenging sounds."
            },
            'Neuroticism': {
                'high': "Music serves as an important emotional outlet and processing tool for you.",
                'medium': "Your music reflects a balanced range of emotional experiences.",
                'low': "You tend to prefer stable, positive music that maintains good vibes."
            }
        }
        
        st.subheader("Personality Insights")
        
        for trait, score in predictions.items():
            if score >= 3.5:
                category = 'high'
            elif score <= 2.5:
                category = 'low'
            else:
                category = 'medium'
            
            with st.expander(f"{trait}: {score:.1f}/5.0"):
                st.write(trait_descriptions[trait][category])
    
    # Music analysis details
    with st.expander("Detailed Music Analysis"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Listening Preferences")
            st.write(f"**Mainstream preference:** {features['mainstream_preference']:.2f}")
            st.write(f"**Underground preference:** {features['underground_preference']:.2f}")
            st.write(f"**Average popularity:** {features['avg_popularity']:.2f}")
            st.write(f"**Explicit content ratio:** {features['explicit_content_ratio']:.2f}")
        
        with col2:
            st.subheader("Audio Characteristics")
            st.write(f"**Energy:** {features['energy']:.2f}")
            st.write(f"**Danceability:** {features['danceability']:.2f}")
            st.write(f"**Valence (positivity):** {features['valence']:.2f}")
            st.write(f"**Acousticness:** {features['acousticness']:.2f}")
        
        # Genre preferences
        if data['genres']:
            st.subheader("Genre Preferences")
            genre_prefs = {
                'Electronic/Dance': features['electronic_preference'],
                'Rock/Alternative': features['rock_preference'],
                'Pop/Mainstream': features['pop_preference'],
                'Classical': features['classical_preference'],
                'Jazz/Blues': features['jazz_preference'],
                'Folk/Country': features['folk_preference'],
                'Hip-Hop/Rap': features['hiphop_preference']
            }
            
            genre_df = pd.DataFrame(list(genre_prefs.items()), columns=['Genre', 'Preference'])
            st.bar_chart(genre_df.set_index('Genre'))
        
        # Sample tracks
        if data['tracks']:
            st.subheader("Sample of Your Music")
            sample_tracks = data['tracks'][:10]
            track_df = pd.DataFrame([
                {
                    'Track': track['name'],
                    'Artist': track['artists'][0]['name'],
                    'Popularity': track.get('popularity', 0)
                }
                for track in sample_tracks
            ])
            st.dataframe(track_df, hide_index=True)

def main():
    st.title("Spotify Musical Personality Predictor")
    st.markdown("Discover your Big Five personality traits through your music taste!")
    
    # Session info in sidebar
    with st.sidebar:
        st.header("Session Information")
        st.write(f"Session ID: {get_unique_session_id()[:8]}...")
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Authenticate with Spotify
    sp = authenticate_spotify()
    
    if sp is None:
        st.info("Please complete Spotify authentication to continue.")
        return
    
    # Verify user identity
    user_info = verify_user_identity(sp)
    
    if user_info is None:
        st.error("Could not verify your Spotify account. Please try logging in again.")
        if st.button("Try Again"):
            clear_authentication_state()
            st.rerun()
        return
    
    # Main analysis button
    if st.button("Analyze My Musical Personality", type="primary", use_container_width=True):
        
        with st.spinner("Collecting your Spotify data..."):
            data = collect_spotify_data(sp, user_info)
        
        if data is None:
            st.error("Could not collect your Spotify data. Please check your account permissions.")
            return
        
        with st.spinner("Analyzing your music preferences..."):
            features = extract_music_features(data)
        
        if features is None:
            st.error("Could not analyze your music data.")
            return
        
        with st.spinner("Predicting your personality traits..."):
            predictions = predict_personality(features)
        
        if predictions is None:
            st.error("Could not generate personality predictions.")
            return
        
        # Display results
        display_results(data, features, predictions)
        
        # Store results for comparison
        result_key = f"analysis_{get_unique_session_id()}"
        st.session_state[result_key] = {
            'user_info': user_info,
            'predictions': predictions,
            'fingerprint': data['fingerprint'],
            'timestamp': datetime.now().isoformat()
        }
    
    # Show comparison if multiple analyses exist
    analysis_keys = [k for k in st.session_state.keys() if k.startswith('analysis_')]
    
    if len(analysis_keys) > 1:
        st.header("Session Comparison")
        st.write("Multiple analyses detected:")
        
        for key in analysis_keys:
            result = st.session_state[key]
            st.write(f"**{result['user_info']['display_name']}** ({result['user_info']['id']}) - Fingerprint: `{result['fingerprint']}`")
        
        # Check for uniqueness
        fingerprints = [st.session_state[key]['fingerprint'] for key in analysis_keys]
        if len(set(fingerprints)) == len(fingerprints):
            st.success("All analyses have unique data fingerprints - isolation is working!")
        else:
            st.error("Duplicate fingerprints detected - there may be an isolation issue.")

if __name__ == "__main__":
    main()