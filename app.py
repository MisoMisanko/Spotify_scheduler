# app.py - Working Spotify Personality Predictor
import os
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from production_predictor import PersonalityPredictor
import requests

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
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

# Initialize the ONLY model we use
@st.cache_resource
def load_personality_predictor():
    """Load the working production model"""
    try:
        predictor = PersonalityPredictor()
        st.sidebar.success("✅ Production model loaded successfully")
        return predictor
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
        return None

# Keep your working authentication (unchanged)
def ensure_spotify_client():
    """Fixed authentication using session-specific cache files"""
    
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        st.error("Missing Spotify credentials. Please set environment variables.")
        st.stop()
    
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
                
                # Verify user identity
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
    st.info("Please log in with Spotify to analyze your music")
    st.markdown(f"[🎵 Log in with Spotify]({login_url})")
    st.stop()

def get_lastfm_genres(artist_name):
    """Get genres from Last.fm and normalize them"""
    if not LASTFM_API_KEY:
        return []
    
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'artist.gettoptags',
            'artist': artist_name,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'toptags' in data and 'tag' in data['toptags']:
                tags = [tag['name'].lower() for tag in data['toptags']['tag']]
                return normalize_genres(tags)
    except:
        pass
    return []

def normalize_genres(raw_genres):
    """Convert raw genre tags to the 19 genres our model expects"""
    
    # Map various genre terms to our model's expected genres
    genre_mapping = {
        'Dance': ['dance', 'electronic', 'edm', 'house', 'techno', 'disco', 'club'],
        'Rock': ['rock', 'hard rock', 'classic rock', 'indie rock', 'alternative rock'],
        'Pop': ['pop', 'mainstream', 'chart', 'radio', 'teen pop', 'dance pop'],
        'Metal or Hardrock': ['metal', 'heavy metal', 'death metal', 'black metal', 'hardrock'],
        'Classical music': ['classical', 'orchestra', 'symphony', 'baroque', 'romantic'],
        'Jazz': ['jazz', 'blues', 'swing', 'bebop', 'smooth jazz'],
        'Folk': ['folk', 'acoustic', 'singer-songwriter', 'indie folk'],
        'Country': ['country', 'bluegrass', 'americana', 'alt-country'],
        'Hiphop, Rap': ['hip-hop', 'rap', 'hip hop', 'trap', 'gangsta rap'],
        'Punk': ['punk', 'punk rock', 'hardcore punk', 'pop punk'],
        'Alternative': ['alternative', 'indie', 'grunge', 'shoegaze'],
        'Latino': ['latin', 'latino', 'reggaeton', 'salsa', 'bachata', 'cumbia'],
        'Reggae, Ska': ['reggae', 'ska', 'dub', 'roots reggae'],
        'Opera': ['opera', 'operatic', 'vocal', 'art song'],
        'Musical': ['musical', 'theatre', 'broadway', 'show tunes'],
        'Techno, Trance': ['techno', 'trance', 'progressive', 'ambient electronic']
    }
    
    normalized = []
    raw_text = ' '.join(raw_genres).lower()
    
    for model_genre, keywords in genre_mapping.items():
        if any(keyword in raw_text for keyword in keywords):
            normalized.append(model_genre)
    
    return normalized

def collect_spotify_data(sp):
    """Collect and process Spotify data with robust error handling"""
    
    progress = st.progress(0)
    status = st.empty()
    
    # Step 1: Get tracks
    status.text("🎵 Fetching your music library...")
    all_tracks = []
    
    for time_range in ['short_term', 'medium_term', 'long_term']:
        try:
            tracks = sp.current_user_top_tracks(limit=50, time_range=time_range)
            all_tracks.extend(tracks['items'])
        except Exception as e:
            st.write(f"Could not get {time_range} tracks: {e}")
    
    # Get recent tracks
    try:
        recent = sp.current_user_recently_played(limit=50)
        all_tracks.extend([item['track'] for item in recent['items']])
    except Exception as e:
        st.write(f"Could not get recent tracks: {e}")
    
    if not all_tracks:
        st.error("No tracks found!")
        return None
    
    # Remove duplicates
    unique_tracks = []
    seen_ids = set()
    for track in all_tracks:
        if track['id'] and track['id'] not in seen_ids:
            unique_tracks.append(track)
            seen_ids.add(track['id'])
    
    st.write(f"✅ Found {len(unique_tracks)} unique tracks")
    progress.progress(0.3)
    
    # Step 2: Get audio features with robust error handling
    status.text("🎼 Analyzing audio characteristics...")
    track_ids = [track['id'] for track in unique_tracks]
    all_audio_features = []
    
    # Try multiple batch sizes if needed
    batch_sizes = [100, 50, 20, 10]
    
    for batch_size in batch_sizes:
        if not track_ids:
            break
            
        remaining_ids = track_ids.copy()
        
        for i in range(0, len(remaining_ids), batch_size):
            batch = remaining_ids[i:i+batch_size]
            try:
                features = sp.audio_features(batch)
                if features:
                    valid_features = [f for f in features if f]
                    all_audio_features.extend(valid_features)
                    # Remove successful IDs
                    successful_ids = [f['id'] for f in valid_features]
                    track_ids = [tid for tid in track_ids if tid not in successful_ids]
                time.sleep(0.1)
            except Exception as e:
                if "403" in str(e) or "429" in str(e):
                    st.warning(f"API limit reached with batch size {batch_size}, trying smaller batches...")
                    time.sleep(2)
                    break
                else:
                    continue
        
        if not track_ids:  # All successful
            break
        elif batch_size == 10:  # Last attempt failed
            st.warning(f"Could not get audio features for {len(track_ids)} tracks - will use fallback estimation")
    
    progress.progress(0.6)
    
    # Step 3: Get and normalize genres
    status.text("🏷️ Identifying genres...")
    all_genres = []
    
    # Get Spotify genres
    artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks[:100]]))
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            artists = sp.artists(batch)
            for artist in artists['artists']:
                raw_genres = artist.get('genres', [])
                all_genres.extend(normalize_genres(raw_genres))
            time.sleep(0.1)
        except Exception as e:
            st.write(f"Error getting artist info: {e}")
    
    # Enrich with Last.fm (sample to avoid rate limits)
    if LASTFM_API_KEY:
        sample_artists = unique_tracks[:20]  # Limit to 20 to avoid timeout
        for track in sample_artists:
            artist_name = track['artists'][0]['name']
            lastfm_genres = get_lastfm_genres(artist_name)
            all_genres.extend(lastfm_genres)
            time.sleep(0.1)
    
    progress.progress(1.0)
    status.text("✅ Data collection complete!")
    
    return {
        'tracks': unique_tracks,
        'audio_features': all_audio_features,
        'genres': all_genres
    }

def create_fallback_audio_features(tracks, genres):
    """Create estimated audio features when Spotify API fails"""
    st.info("Creating estimated audio features from track metadata...")
    
    fallback_features = []
    for track in tracks:
        popularity = track.get('popularity', 50)
        track_name = track['name'].lower()
        artist_name = track['artists'][0]['name'].lower()
        
        # Estimate features based on genre and metadata
        estimated = {
            'danceability': 0.5,
            'energy': 0.5,
            'valence': 0.5,
            'acousticness': 0.3,
            'instrumentalness': 0.1,
            'loudness': -10,
            'speechiness': 0.1
        }
        
        # Adjust based on genres
        genre_text = ' '.join(genres).lower()
        if any(term in genre_text for term in ['dance', 'electronic', 'edm', 'house']):
            estimated['danceability'] = 0.8
            estimated['energy'] = 0.8
        if any(term in genre_text for term in ['rock', 'metal', 'punk']):
            estimated['energy'] = 0.9
            estimated['loudness'] = -5
        if any(term in genre_text for term in ['acoustic', 'folk', 'classical']):
            estimated['acousticness'] = 0.8
            estimated['energy'] = 0.3
        if any(term in genre_text for term in ['hip-hop', 'rap']):
            estimated['speechiness'] = 0.4
            estimated['energy'] = 0.7
        
        # Adjust based on track/artist names
        if any(word in track_name for word in ['love', 'happy', 'good']):
            estimated['valence'] = 0.7
        if any(word in track_name for word in ['sad', 'dark', 'pain']):
            estimated['valence'] = 0.3
            
        fallback_features.append(estimated)
    
    return fallback_features

def predict_personality(music_data, predictor):
    """Use our working model to predict personality with robust fallback"""
    
    # Try to use real audio features first, then fallback
    if music_data['audio_features']:
        st.success(f"Using {len(music_data['audio_features'])} real audio features")
        audio_features = music_data['audio_features']
    else:
        st.warning("No audio features from Spotify - creating estimates from metadata")
        audio_features = create_fallback_audio_features(music_data['tracks'], music_data['genres'])
    
    if not audio_features:
        st.error("Could not create any audio features")
        return None
    
    # Calculate average audio features
    audio_df = pd.DataFrame(audio_features)
    
    spotify_features = {
        'danceability': audio_df['danceability'].mean(),
        'energy': audio_df['energy'].mean(), 
        'valence': audio_df['valence'].mean(),
        'acousticness': audio_df['acousticness'].mean(),
        'instrumentalness': audio_df['instrumentalness'].mean(),
        'loudness': audio_df['loudness'].mean(),
        'speechiness': audio_df['speechiness'].mean()
    }
    
    # Show music profile
    st.write("### 🎵 Your Music Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Danceability", f"{spotify_features['danceability']:.2f}")
        st.metric("Energy", f"{spotify_features['energy']:.2f}")
    with col2:
        st.metric("Valence", f"{spotify_features['valence']:.2f}")
        st.metric("Acousticness", f"{spotify_features['acousticness']:.2f}")
    with col3:
        st.metric("Speechiness", f"{spotify_features['speechiness']:.2f}")
        st.metric("Instrumentalness", f"{spotify_features['instrumentalness']:.2f}")
    with col4:
        st.metric("Loudness", f"{spotify_features['loudness']:.1f} dB")
        unique_genres = len(set(music_data['genres']))
        st.metric("Unique Genres", unique_genres)
    
    # Get predictions
    try:
        predictions = predictor.predict_from_spotify_features(spotify_features)
        
        if 'error' in predictions:
            st.error(f"Prediction failed: {predictions['error']}")
            return None
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def display_results(predictions, predictor):
    """Display personality results"""
    
    st.header("🎭 Your Musical Personality")
    
    # Radar chart
    traits = list(predictions.keys())
    scores = list(predictions.values())
    
    fig = go.Figure()
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
    
    # Individual traits
    trait_descriptions = {
        'Openness': 'Creativity, curiosity, and openness to new experiences',
        'Conscientiousness': 'Organization, discipline, and goal-oriented behavior', 
        'Extraversion': 'Sociability, energy, and positive emotions',
        'Agreeableness': 'Compassion, cooperation, and trust in others',
        'Neuroticism': 'Emotional instability and tendency toward negative emotions'
    }
    
    for trait, score in predictions.items():
        level = "High" if score >= 3.5 else "Low" if score <= 2.5 else "Moderate"
        
        with st.expander(f"**{trait}**: {score:.2f}/5.0 ({level})"):
            st.write(f"**What this measures:** {trait_descriptions.get(trait, 'Personality dimension')}")
            st.progress(score / 5.0)
            
            if score >= 3.5:
                st.info(f"You score high on {trait}")
            elif score <= 2.5:
                st.info(f"You score low on {trait}")
            else:
                st.info(f"You score moderately on {trait}")

def main():
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits from your music!")
    
    # Load our working model
    predictor = load_personality_predictor()
    if not predictor:
        st.error("Could not load personality model. Check if production_personality_models.pkl exists in models/ folder.")
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("How It Works")
        st.write("""
        1. **Analyzes** your Spotify listening history
        2. **Extracts** audio features and genres  
        3. **Maps** them to personality patterns
        4. **Predicts** your Big Five traits
        
        Based on research with 1,010 participants.
        """)
        
        if LASTFM_API_KEY:
            st.success("✅ Last.fm integration enabled")
        else:
            st.info("ℹ️ Using Spotify data only")
    
    # Authentication
    sp = ensure_spotify_client()
    
    # Show logged in user
    if hasattr(st.session_state, 'user') and st.session_state.user:
        user = st.session_state.user
        st.success(f"🎧 Connected as: **{user.get('display_name', 'Unknown')}**")
    
    # Main analysis button
    if st.button("🚀 Analyze My Musical Personality", type="primary", use_container_width=True):
        
        # Collect data
        with st.spinner("Collecting your music data..."):
            music_data = collect_spotify_data(sp)
        
        if music_data:
            # Predict personality
            with st.spinner("Analyzing personality patterns..."):
                predictions = predict_personality(music_data, predictor)
            
            if predictions:
                # Display results
                display_results(predictions, predictor)
                
                # Show what genres were found
                if music_data['genres']:
                    st.subheader("🎼 Genres Detected")
                    unique_genres = list(set(music_data['genres']))
                    if unique_genres:
                        st.write(", ".join(unique_genres))
                    else:
                        st.write("No specific genres identified")
                
                # Model info
                with st.expander("About the Model"):
                    model_info = predictor.get_model_info()
                    st.write("**Model Performance:**")
                    if model_info and 'performance' in model_info:
                        for trait, perf in model_info['performance'].items():
                            st.write(f"- {trait}: {perf['accuracy']} accuracy (R² = {perf['r2_score']})")
                    
                    st.write("""
                    **How it works:**
                    - Trained on genre preferences → personality data
                    - Your Spotify audio features are mapped to genre estimates  
                    - Personality predicted from estimated genre preferences
                    - Based on academic research linking music taste to personality
                    """)
        else:
            st.error("Could not collect your music data. Make sure you have Spotify listening history.")

if __name__ == "__main__":
    main()