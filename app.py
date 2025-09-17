# app.py - Fixed Spotify Personality Predictor - Genre Focused
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
from collections import Counter, defaultdict
import asyncio
from datetime import datetime, timedelta

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

class GenreEnricher:
    """Robust genre enrichment using multiple sources"""
    
    def __init__(self, lastfm_key=None):
        self.lastfm_key = lastfm_key
        self.artist_cache = {}
        
        # Expanded genre mapping for normalization
        self.genre_mapping = {
            'Dance': ['dance', 'electronic', 'edm', 'house', 'techno', 'disco', 'club', 'electro', 'electronica'],
            'Rock': ['rock', 'hard rock', 'classic rock', 'indie rock', 'alternative rock', 'rock and roll', 'soft rock'],
            'Pop': ['pop', 'mainstream', 'chart', 'radio', 'teen pop', 'dance pop', 'synthpop', 'electropop'],
            'Metal or Hardrock': ['metal', 'heavy metal', 'death metal', 'black metal', 'hardrock', 'metalcore', 'doom metal'],
            'Classical music': ['classical', 'orchestra', 'symphony', 'baroque', 'romantic', 'contemporary classical', 'chamber music'],
            'Swing, Jazz': ['jazz', 'blues', 'swing', 'bebop', 'smooth jazz', 'acid jazz', 'fusion', 'big band'],
            'Folk': ['folk', 'acoustic', 'singer-songwriter', 'indie folk', 'folk rock', 'traditional folk'],
            'Country': ['country', 'bluegrass', 'americana', 'alt-country', 'outlaw country', 'contemporary country'],
            'Hiphop, Rap': ['hip-hop', 'rap', 'hip hop', 'trap', 'gangsta rap', 'conscious rap', 'old school rap'],
            'Punk': ['punk', 'punk rock', 'hardcore punk', 'pop punk', 'post-punk', 'ska punk'],
            'Alternative': ['alternative', 'indie', 'grunge', 'shoegaze', 'britpop', 'post-rock', 'art rock'],
            'Latino': ['latin', 'latino', 'reggaeton', 'salsa', 'bachata', 'cumbia', 'bossa nova', 'tango', 'flamenco'],
            'Reggae, Ska': ['reggae', 'ska', 'dub', 'roots reggae', 'dancehall', 'ragga'],
            'Opera': ['opera', 'operatic', 'vocal', 'art song', 'classical vocal'],
            'Musical': ['musical', 'theatre', 'broadway', 'show tunes', 'soundtrack'],
            'Techno, Trance': ['techno', 'trance', 'progressive', 'ambient electronic', 'minimal techno', 'psytrance'],
            'Rock n roll': ['rock n roll', 'rockabilly', 'doo-wop', '50s rock', 'early rock'],
            'Slow songs or fast songs': ['ballad', 'slow', 'mellow', 'chill', 'downtempo'],
            'Music': ['instrumental', 'world music', 'new age', 'ambient', 'experimental']
        }
    
    def get_lastfm_tags(self, artist_name):
        """Get tags from Last.fm with better error handling"""
        if not self.lastfm_key or artist_name in self.artist_cache:
            return self.artist_cache.get(artist_name, [])
        
        try:
            url = "http://ws.audioscrobbler.com/2.0/"
            params = {
                'method': 'artist.gettoptags',
                'artist': artist_name,
                'api_key': self.lastfm_key,
                'format': 'json',
                'limit': 10  # Get more tags
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'toptags' in data and 'tag' in data['toptags']:
                    tags = [tag['name'].lower().strip() for tag in data['toptags']['tag']]
                    normalized = self.normalize_genres(tags)
                    self.artist_cache[artist_name] = normalized
                    return normalized
        except Exception as e:
            st.write(f"Last.fm error for {artist_name}: {e}")
        
        self.artist_cache[artist_name] = []
        return []
    
    def get_musicbrainz_tags(self, artist_name):
        """Get tags from MusicBrainz as fallback"""
        try:
            # Search for artist
            search_url = f"https://musicbrainz.org/ws/2/artist/?query={artist_name}&fmt=json&limit=1"
            headers = {'User-Agent': 'SpotifyPersonalityApp/1.0 (research@university.edu)'}
            
            response = requests.get(search_url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                artists = data.get('artists', [])
                if artists:
                    artist_id = artists[0]['id']
                    
                    # Get detailed info with tags
                    detail_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=tags&fmt=json"
                    time.sleep(1.1)  # Respect rate limit
                    
                    detail_response = requests.get(detail_url, headers=headers, timeout=5)
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        tags = detail_data.get('tags', [])
                        tag_names = [tag['name'].lower().strip() for tag in tags if tag.get('count', 0) > 0]
                        return self.normalize_genres(tag_names)
        except Exception as e:
            st.write(f"MusicBrainz error for {artist_name}: {e}")
        
        return []
    
    def normalize_genres(self, raw_genres):
        """Convert raw genre tags to the 19 genres our model expects"""
        normalized = []
        raw_text = ' '.join(raw_genres).lower()
        
        for model_genre, keywords in self.genre_mapping.items():
            if any(keyword in raw_text for keyword in keywords):
                normalized.append(model_genre)
        
        return list(set(normalized))  # Remove duplicates
    
    def enrich_artist_genres(self, artists_data):
        """Enrich artist data with genres from multiple sources"""
        enriched_genres = []
        
        for artist in artists_data:
            artist_name = artist['name']
            
            # Start with Spotify genres
            spotify_genres = self.normalize_genres(artist.get('genres', []))
            all_genres = spotify_genres.copy()
            
            # Add Last.fm tags if available
            if self.lastfm_key:
                lastfm_tags = self.get_lastfm_tags(artist_name)
                all_genres.extend(lastfm_tags)
                time.sleep(0.2)  # Rate limiting
            
            # Use MusicBrainz as fallback if we have few genres
            if len(set(all_genres)) < 2:
                mb_tags = self.get_musicbrainz_tags(artist_name)
                all_genres.extend(mb_tags)
            
            enriched_genres.extend(all_genres)
        
        return enriched_genres

def collect_listening_data(sp):
    """Collect listening data with temporal analysis"""
    
    progress = st.progress(0)
    status = st.empty()
    
    listening_data = {
        'short_term': [],    # Last 4 weeks
        'medium_term': [],   # Last 6 months  
        'long_term': [],     # Last 2 years
        'recent': []         # Last 50 tracks
    }
    
    # Step 1: Get tracks from different time periods
    status.text("🎵 Fetching listening history across time periods...")
    
    time_ranges = ['short_term', 'medium_term', 'long_term']
    for i, time_range in enumerate(time_ranges):
        try:
            tracks_response = sp.current_user_top_tracks(limit=50, time_range=time_range)
            tracks = tracks_response['items']
            listening_data[time_range] = tracks
            st.write(f"✅ {time_range}: {len(tracks)} tracks")
        except Exception as e:
            st.write(f"Could not get {time_range} tracks: {e}")
        
        progress.progress((i + 1) / 5)
        time.sleep(0.1)
    
    # Get recent tracks with timestamps
    try:
        recent_response = sp.current_user_recently_played(limit=50)
        listening_data['recent'] = [item['track'] for item in recent_response['items']]
        st.write(f"✅ Recent: {len(listening_data['recent'])} tracks")
    except Exception as e:
        st.write(f"Could not get recent tracks: {e}")
    
    progress.progress(0.8)
    
    # Step 2: Extract unique artists across all periods
    status.text("🎤 Collecting artist information...")
    
    all_tracks = []
    for period_tracks in listening_data.values():
        all_tracks.extend(period_tracks)
    
    # Get unique artists
    unique_artists = {}
    for track in all_tracks:
        if track and track.get('artists'):
            artist = track['artists'][0]
            artist_id = artist['id']
            if artist_id and artist_id not in unique_artists:
                unique_artists[artist_id] = {
                    'id': artist_id,
                    'name': artist['name']
                }
    
    st.write(f"✅ Found {len(unique_artists)} unique artists")
    
    # Step 3: Get artist details with genres (batch processing)
    artist_details = []
    artist_ids = list(unique_artists.keys())
    
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            artists_response = sp.artists(batch)
            artist_details.extend(artists_response['artists'])
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            st.write(f"Error getting artist batch: {e}")
    
    progress.progress(1.0)
    status.text("✅ Data collection complete!")
    
    return {
        'listening_periods': listening_data,
        'artists': artist_details,
        'unique_tracks': len(all_tracks)
    }

def analyze_listening_consistency(listening_data, genres_by_period):
    """Analyze consistency in listening patterns over time"""
    
    consistency_metrics = {}
    
    # Calculate genre distributions for each period
    distributions = {}
    for period, genres in genres_by_period.items():
        if genres:
            genre_counts = Counter(genres)
            total = sum(genre_counts.values())
            distributions[period] = {genre: count/total for genre, count in genre_counts.items()}
        else:
            distributions[period] = {}
    
    # Compare short vs long term consistency
    if distributions['short_term'] and distributions['long_term']:
        # Calculate overlap
        short_genres = set(distributions['short_term'].keys())
        long_genres = set(distributions['long_term'].keys())
        
        overlap = len(short_genres & long_genres)
        total_unique = len(short_genres | long_genres)
        
        consistency_metrics['genre_stability'] = overlap / total_unique if total_unique > 0 else 0
        
        # Calculate correlation of preferences
        common_genres = short_genres & long_genres
        if len(common_genres) >= 3:
            short_prefs = [distributions['short_term'][g] for g in common_genres]
            long_prefs = [distributions['long_term'][g] for g in common_genres]
            correlation = np.corrcoef(short_prefs, long_prefs)[0,1]
            consistency_metrics['preference_correlation'] = correlation if not np.isnan(correlation) else 0
        else:
            consistency_metrics['preference_correlation'] = 0
    
    # Calculate diversity metrics
    for period, dist in distributions.items():
        if dist:
            # Shannon diversity index
            probs = list(dist.values())
            diversity = -sum(p * np.log(p) for p in probs if p > 0)
            consistency_metrics[f'{period}_diversity'] = diversity
    
    return consistency_metrics

def predict_personality_from_genres(genre_data, consistency_metrics, predictor):
    """Predict personality directly from genre preferences with enhanced sensitivity"""
    
    # Calculate overall genre preferences (weighted by recency)
    all_genres = []
    
    # Weight recent listening more heavily
    weights = {
        'short_term': 0.4,   # Recent preferences matter most
        'medium_term': 0.3,  # Medium term shows patterns
        'long_term': 0.2,    # Long term shows core taste
        'recent': 0.1        # Very recent might be situational
    }
    
    weighted_genre_counts = defaultdict(float)
    total_weight = 0
    
    for period, genres in genre_data.items():
        weight = weights.get(period, 0.1)
        for genre in genres:
            weighted_genre_counts[genre] += weight
            total_weight += weight
    
    # Get all expected genres for the model
    expected_genres = [
        'Music', 'Slow songs or fast songs', 'Dance', 'Folk', 'Country',
        'Classical music', 'Musical', 'Pop', 'Rock', 'Metal or Hardrock',
        'Punk', 'Hiphop, Rap', 'Reggae, Ska', 'Swing, Jazz', 'Rock n roll',
        'Alternative', 'Latino', 'Techno, Trance', 'Opera'
    ]
    
    # Calculate preferences with MORE EXTREME scaling (1-5 scale)
    max_count = max(weighted_genre_counts.values()) if weighted_genre_counts else 1
    genre_preferences = {}
    
    for genre in expected_genres:
        if genre in weighted_genre_counts:
            # More aggressive scaling - amplify differences
            raw_pref = weighted_genre_counts[genre] / max_count
            # Use exponential scaling to create more extreme preferences
            if raw_pref > 0.3:  # Strong preference
                preference = 3.5 + (raw_pref * 1.5)  # Scale 3.5-5.0
            elif raw_pref > 0.1:  # Moderate preference  
                preference = 2.5 + (raw_pref * 2)    # Scale 2.5-3.5
            else:  # Low/no preference
                preference = 1.0 + (raw_pref * 1.5)  # Scale 1.0-2.5
        else:
            # Lower neutral for unheard genres to create more contrast
            preference = 2.5
        
        genre_preferences[genre] = min(5.0, max(1.0, preference))
    
    # Get base predictions
    base_predictions = predictor.predict_from_genres(genre_preferences)
    
    if 'error' in base_predictions:
        return base_predictions
    
    # MUCH MORE AGGRESSIVE adjustments based on musical patterns
    adjusted_predictions = base_predictions.copy()
    
    # Genre-specific personality mappings (based on research)
    alternative_score = weighted_genre_counts.get('Alternative', 0)
    pop_score = weighted_genre_counts.get('Pop', 0)
    rock_score = weighted_genre_counts.get('Rock', 0)
    classical_score = weighted_genre_counts.get('Classical music', 0)
    punk_score = weighted_genre_counts.get('Punk', 0)
    hiphop_score = weighted_genre_counts.get('Hiphop, Rap', 0)
    
    # OPENNESS: Alternative/experimental music = higher openness
    openness_boost = 0
    if alternative_score > 0.2 * max_count:  # Significant alternative listening
        openness_boost += 0.8
    if classical_score > 0.1 * max_count:
        openness_boost += 0.6
    if punk_score > 0.1 * max_count:
        openness_boost += 0.4
    
    # Diversity bonus
    if 'short_term_diversity' in consistency_metrics:
        diversity = consistency_metrics['short_term_diversity']
        openness_boost += min(0.7, diversity / 3)  # Up to 0.7 boost
    
    # Less consistency = more openness
    if 'preference_correlation' in consistency_metrics:
        correlation = consistency_metrics['preference_correlation']
        openness_boost += (1 - correlation) * 0.6
    
    adjusted_predictions['Openness'] = min(5.0, max(1.0, 
        adjusted_predictions['Openness'] + openness_boost))
    
    # CONSCIENTIOUSNESS: Pop music = higher conscientiousness, punk = lower
    conscientiousness_adj = 0
    if pop_score > 0.3 * max_count:  # Lots of mainstream pop
        conscientiousness_adj += 0.6
    if punk_score > 0.1 * max_count:  # Punk = anti-establishment
        conscientiousness_adj -= 0.5
    
    # High stability = higher conscientiousness  
    if 'genre_stability' in consistency_metrics:
        stability = consistency_metrics['genre_stability']
        conscientiousness_adj += stability * 0.8  # Up to 0.8 boost
    
    adjusted_predictions['Conscientiousness'] = min(5.0, max(1.0,
        adjusted_predictions['Conscientiousness'] + conscientiousness_adj))
    
    # EXTRAVERSION: Dance/Pop = higher, Alternative/Classical = lower
    extraversion_adj = 0
    dance_score = weighted_genre_counts.get('Dance', 0)
    if dance_score > 0.1 * max_count or pop_score > 0.3 * max_count:
        extraversion_adj += 0.7
    if alternative_score > 0.3 * max_count or classical_score > 0.1 * max_count:
        extraversion_adj -= 0.4
    
    adjusted_predictions['Extraversion'] = min(5.0, max(1.0,
        adjusted_predictions['Extraversion'] + extraversion_adj))
    
    # NEUROTICISM: Alternative/emotional music = higher neuroticism
    neuroticism_adj = 0
    if alternative_score > 0.3 * max_count:
        neuroticism_adj += 0.6
    if punk_score > 0.1 * max_count:
        neuroticism_adj += 0.5
    # Low pop = higher neuroticism (less mainstream appeal)
    if pop_score < 0.1 * max_count:
        neuroticism_adj += 0.4
    
    adjusted_predictions['Neuroticism'] = min(5.0, max(1.0,
        adjusted_predictions['Neuroticism'] + neuroticism_adj))
    
    # AGREEABLENESS: Pop = higher, punk/metal = lower
    agreeableness_adj = 0
    metal_score = weighted_genre_counts.get('Metal or Hardrock', 0)
    if pop_score > 0.2 * max_count:
        agreeableness_adj += 0.5
    if punk_score > 0.1 * max_count or metal_score > 0.1 * max_count:
        agreeableness_adj -= 0.6
    
    adjusted_predictions['Agreeableness'] = min(5.0, max(1.0,
        adjusted_predictions['Agreeableness'] + agreeableness_adj))
    
    # Round results
    for trait in adjusted_predictions:
        adjusted_predictions[trait] = round(adjusted_predictions[trait], 2)
    
    return adjusted_predictions, consistency_metrics

def display_results(predictions, consistency_metrics, genre_data, predictor):
    """Display comprehensive personality results with listening analysis"""
    
    st.header("🎭 Your Musical Personality")
    
    # Show consistency insights first
    if consistency_metrics:
        st.subheader("🔄 Your Listening Patterns")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stability = consistency_metrics.get('genre_stability', 0)
            st.metric("Genre Stability", f"{stability:.2%}")
            
        with col2:
            correlation = consistency_metrics.get('preference_correlation', 0)
            st.metric("Preference Consistency", f"{correlation:.2f}")
            
        with col3:
            diversity = consistency_metrics.get('short_term_diversity', 0)
            st.metric("Recent Diversity", f"{diversity:.2f}")
        
        # Consistency insights
        with st.expander("What does this mean?"):
            st.write(f"""
            **Genre Stability ({stability:.1%})**: How much your favorite genres overlap between recent and long-term listening.
            
            **Preference Consistency ({correlation:.2f})**: How similar your current preferences are to your historical patterns.
            
            **Recent Diversity ({diversity:.2f})**: How varied your recent listening has been across genres.
            
            These patterns influenced your personality predictions, especially Openness and Conscientiousness.
            """)
    
    # Radar chart
    if isinstance(predictions, dict) and 'error' not in predictions:
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
        
        # Individual traits with explanations
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
                
                # Show how consistency affected this trait
                if trait == 'Openness' and consistency_metrics:
                    corr = consistency_metrics.get('preference_correlation', 0)
                    div = consistency_metrics.get('short_term_diversity', 0)
                    st.info(f"Influenced by listening diversity ({div:.2f}) and consistency ({corr:.2f})")
                elif trait == 'Conscientiousness' and consistency_metrics:
                    stability = consistency_metrics.get('genre_stability', 0)
                    st.info(f"Influenced by genre stability over time ({stability:.1%})")
    
    # Show genre analysis
    st.subheader("🎼 Your Musical Taste Analysis")
    
    # Combine all genres for analysis
    all_found_genres = []
    for period_genres in genre_data.values():
        all_found_genres.extend(period_genres)
    
    if all_found_genres:
        genre_counts = Counter(all_found_genres)
        top_genres = genre_counts.most_common(10)
        
        st.write("**Top Genres Detected:**")
        st.write(", ".join([f"{genre} ({count})" for genre, count in top_genres]))
        
        # Show temporal differences
        st.write("**Genre Evolution:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Favorites:**")
            recent_genres = Counter(genre_data.get('short_term', []))
            for genre, count in recent_genres.most_common(5):
                st.write(f"- {genre}: {count}")
        
        with col2:
            st.write("**Long-term Favorites:**")
            longterm_genres = Counter(genre_data.get('long_term', []))
            for genre, count in longterm_genres.most_common(5):
                st.write(f"- {genre}: {count}")
    else:
        st.warning("No genres detected. This might affect prediction accuracy.")

def main():
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits from your music listening patterns!")
    
    # Load model
    predictor = load_personality_predictor()
    if not predictor:
        st.error("Could not load personality model.")
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("Enhanced Analysis")
        st.write("""
        **New Features:**
        - Direct genre-based prediction
        - Listening consistency analysis
        - Temporal pattern recognition
        - Multi-source genre enrichment
        
        **Data Sources:**
        - Spotify artist genres
        - Last.fm community tags
        - MusicBrainz metadata
        """)
        
        if LASTFM_API_KEY:
            st.success("✅ Last.fm integration enabled")
        else:
            st.info("ℹ️ Using Spotify + MusicBrainz only")
    
    # Authentication
    sp = ensure_spotify_client()
    
    # Show logged in user
    if hasattr(st.session_state, 'user') and st.session_state.user:
        user = st.session_state.user
        st.success(f"🎧 Connected as: **{user.get('display_name', 'Unknown')}**")
    
    # Main analysis button
    if st.button("🚀 Analyze My Musical Personality", type="primary", use_container_width=True):
        
        # Collect listening data
        with st.spinner("Collecting your listening history across time periods..."):
            listening_data = collect_listening_data(sp)
        
        if listening_data and listening_data['artists']:
            
            # Enrich with genres
            with st.spinner("Enriching genre data from multiple sources..."):
                enricher = GenreEnricher(LASTFM_API_KEY)
                
                # Process genres for each time period
                genres_by_period = {}
                
                for period, tracks in listening_data['listening_periods'].items():
                    if tracks:
                        # Get artists for this period
                        period_artist_ids = [track['artists'][0]['id'] for track in tracks if track.get('artists')]
                        period_artists = [artist for artist in listening_data['artists'] if artist['id'] in period_artist_ids]
                        
                        # Enrich genres for this period
                        period_genres = enricher.enrich_artist_genres(period_artists)
                        genres_by_period[period] = period_genres
                        
                        st.write(f"📊 {period}: {len(period_genres)} genre tags from {len(period_artists)} artists")
            
            # Analyze consistency
            with st.spinner("Analyzing listening patterns and consistency..."):
                consistency_metrics = analyze_listening_consistency(listening_data, genres_by_period)
            
            # Predict personality
            with st.spinner("Predicting personality from musical patterns..."):
                prediction_result = predict_personality_from_genres(genres_by_period, consistency_metrics, predictor)
                
                if isinstance(prediction_result, tuple):
                    predictions, consistency_metrics = prediction_result
                else:
                    predictions = prediction_result
            
            # Display results
            if predictions and 'error' not in predictions:
                display_results(predictions, consistency_metrics, genres_by_period, predictor)
                
                # Model info
                with st.expander("About This Analysis"):
                    st.write("""
                    **How it works:**
                    1. Collects your listening history from different time periods
                    2. Enriches sparse Spotify genre data with Last.fm and MusicBrainz
                    3. Analyzes consistency patterns in your musical preferences
                    4. Predicts personality directly from genre preferences
                    5. Adjusts predictions based on listening consistency patterns
                    
                    **Key improvements:**
                    - No more audio feature conversion (direct genre → personality)
                    - Multi-source genre enrichment for better coverage
                    - Temporal consistency analysis affects Openness and Conscientiousness
                    - Weighted preferences based on listening recency
                    """)
                    
                    model_info = predictor.get_model_info()
                    if model_info and 'performance' in model_info:
                        st.write("**Model Performance:**")
                        for trait, perf in model_info['performance'].items():
                            st.write(f"- {trait}: {perf['accuracy']} accuracy (R² = {perf['r2_score']})")
            else:
                st.error("Could not generate personality predictions. Please try again.")
        else:
            st.error("Could not collect enough listening data. Make sure you have an active Spotify listening history.")

if __name__ == "__main__":
    main()