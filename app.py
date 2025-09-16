# app.py - Public Spotify personality app for any user
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import time

# Page config
st.set_page_config(
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

class SpotifyPersonalityApp:
    def __init__(self):
        # Get credentials from Streamlit secrets (baked into the app)
        try:
            self.client_id = st.secrets["SPOTIFY_CLIENT_ID"]
            self.client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
            self.redirect_uri = st.secrets["SPOTIFY_REDIRECT_URI"]
        except KeyError as e:
            st.error(f"Missing Spotify credential in secrets: {e}")
            st.stop()
        
        # Scope for reading user's music data
        scope = "user-read-recently-played user-top-read user-read-private playlist-read-private user-library-read"
        
        # Initialize Spotify client
        self.sp_oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=scope,
            cache_path=None,  # Don't cache in deployment
            show_dialog=True
        )
        
        self.sp = None
    
    def get_auth_url(self):
        """Get Spotify authorization URL"""
        return self.sp_oauth.get_authorize_url()
    
    def authenticate_user(self, code):
        """Authenticate user with authorization code"""
        try:
            token_info = self.sp_oauth.get_access_token(code)
            if token_info:
                self.sp = spotipy.Spotify(auth=token_info['access_token'])
                return True
        except Exception as e:
            st.error(f"Authentication failed: {e}")
        return False
    
    def get_user_music_data(self, limit=50):
        """Get user's music data with PROPER audio features handling"""
        if not self.sp:
            return None
        
        try:
            # Get user's top tracks from different time periods
            st.write("🎵 Fetching your top tracks...")
            top_tracks_data = {}
            
            for time_range in ['short_term', 'medium_term', 'long_term']:
                try:
                    tracks = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)
                    top_tracks_data[time_range] = tracks['items']
                    st.write(f"✅ Got {len(tracks['items'])} tracks from {time_range.replace('_', ' ')}")
                except Exception as e:
                    st.warning(f"Could not get {time_range} tracks: {e}")
                    top_tracks_data[time_range] = []
            
            # Get recent tracks
            st.write("🎵 Fetching your recent listening history...")
            try:
                recent_tracks = self.sp.current_user_recently_played(limit=limit)
                recent_tracks_list = [item['track'] for item in recent_tracks['items']]
                st.write(f"✅ Got {len(recent_tracks_list)} recent tracks")
            except Exception as e:
                st.warning(f"Could not get recent tracks: {e}")
                recent_tracks_list = []
            
            # Combine all tracks
            all_tracks = []
            for tracks in top_tracks_data.values():
                all_tracks.extend(tracks)
            all_tracks.extend(recent_tracks_list)
            
            if not all_tracks:
                st.error("No tracks found. Make sure you have some listening history on Spotify!")
                return None
            
            # Remove duplicates
            seen_ids = set()
            unique_tracks = []
            for track in all_tracks:
                if track['id'] not in seen_ids:
                    unique_tracks.append(track)
                    seen_ids.add(track['id'])
            
            st.write(f"🎯 Processing {len(unique_tracks)} unique tracks...")
            
            # Get audio features - THIS IS THE KEY FIX
            st.write("🎵 Getting audio features...")
            track_ids = [track['id'] for track in unique_tracks if track['id']]
            
            # Spotify audio_features() can handle up to 100 tracks per request
            all_audio_features = []
            for i in range(0, len(track_ids), 100):
                batch_ids = track_ids[i:i+100]
                st.write(f"   Processing batch {i//100 + 1}/{(len(track_ids)-1)//100 + 1}...")
                
                try:
                    batch_features = self.sp.audio_features(batch_ids)
                    # Filter out None responses
                    valid_features = [f for f in batch_features if f is not None]
                    all_audio_features.extend(valid_features)
                    st.write(f"   ✅ Got features for {len(valid_features)} tracks in this batch")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    st.warning(f"Error getting audio features for batch: {e}")
                    continue
            
            if not all_audio_features:
                st.error("❌ Could not get audio features for any tracks!")
                st.info("This might be due to:")
                st.info("• Spotify API rate limiting")
                st.info("• Tracks not having audio features available")
                st.info("• Network connectivity issues")
                return None
            
            st.success(f"✅ Successfully got audio features for {len(all_audio_features)} tracks!")
            
            # Get artists for genre analysis
            st.write("🎤 Getting artist information...")
            artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks if track['artists']]))
            
            all_artists = []
            for i in range(0, len(artist_ids), 50):  # Max 50 artists per request
                batch_ids = artist_ids[i:i+50]
                try:
                    artists_response = self.sp.artists(batch_ids)
                    all_artists.extend(artists_response['artists'])
                    time.sleep(0.1)
                except Exception as e:
                    st.warning(f"Error getting artist info: {e}")
            
            # Collect all genres
            all_genres = []
            for artist in all_artists:
                all_genres.extend(artist.get('genres', []))
            
            st.success(f"✅ Found {len(set(all_genres))} unique genres in your music!")
            
            return {
                'tracks': unique_tracks,
                'audio_features': all_audio_features,
                'artists': all_artists,
                'genres': all_genres
            }
            
        except Exception as e:
            st.error(f"Error fetching music data: {e}")
            return None
    
    def extract_personality_features(self, music_data):
        """Extract features for personality prediction"""
        
        audio_features = music_data['audio_features']
        genres = music_data['genres']
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(audio_features)
        
        # Calculate behavioral features
        features = {
            # Basic audio characteristics
            'energy': df['energy'].mean(),
            'danceability': df['danceability'].mean(), 
            'valence': df['valence'].mean(),
            'acousticness': df['acousticness'].mean(),
            'instrumentalness': df['instrumentalness'].mean(),
            'speechiness': df['speechiness'].mean(),
            'loudness': df['loudness'].mean(),
            'tempo': df['tempo'].mean(),
            'liveness': df['liveness'].mean(),
            
            # Popularity and mainstream preference
            'popularity': df['popularity'].mean(),
            'mainstream_preference': (df['popularity'] > 70).mean(),
            'unpopular_preference': (df['popularity'] < 30).mean(),
            
            # Diversity metrics
            'genre_diversity': len(set(genres)) / max(len(genres), 1),
            'unique_genres_count': len(set(genres)),
            'artist_diversity': len(set([f['id'] for f in audio_features])) / len(audio_features),
            
            # Emotional and energy patterns
            'emotional_variance': df['valence'].std(),
            'energy_variance': df['energy'].std(),
            'tempo_variance': df['tempo'].std(),
            
            # Musical preferences (ratios)
            'positive_music_ratio': (df['valence'] > 0.6).mean(),
            'negative_music_ratio': (df['valence'] < 0.4).mean(),
            'high_energy_ratio': (df['energy'] > 0.7).mean(),
            'low_energy_ratio': (df['energy'] < 0.3).mean(),
            'danceable_ratio': (df['danceability'] > 0.7).mean(),
            'acoustic_ratio': (df['acousticness'] > 0.5).mean(),
            'instrumental_ratio': (df['instrumentalness'] > 0.5).mean(),
            'live_music_ratio': (df['liveness'] > 0.8).mean(),
            'speech_ratio': (df['speechiness'] > 0.66).mean(),
            
            # Complexity indicators
            'musical_complexity': (df['acousticness'] + df['instrumentalness']).mean() / 2,
            'mainstream_avoidance': 1 - (df['popularity'].mean() / 100),
            
            # Key and mode diversity
            'key_diversity': len(df['key'].unique()) / 12,
            'mode_diversity': len(df['mode'].unique()) / 2,
            'time_signature_diversity': len(df['time_signature'].unique()) / 7,
        }
        
        return features
    
    def predict_personality(self, features):
        """Predict Big Five personality traits using research-backed approach"""
        
        predictions = {}
        
        # EXTRAVERSION - Social, energetic, outgoing
        # Research: Extraverts prefer energetic, upbeat, conventional music
        extraversion = (
            features['energy'] * 0.25 +
            features['danceability'] * 0.20 +
            features['valence'] * 0.15 +
            features['high_energy_ratio'] * 0.15 +
            features['danceable_ratio'] * 0.10 +
            (features['loudness'] + 60) / 60 * 0.10 +  # Normalize loudness (-60 to 0)
            features['mainstream_preference'] * 0.05
        )
        predictions['Extraversion'] = np.clip(extraversion * 5, 1, 5)
        
        # OPENNESS - Creative, curious, open to new experiences
        # Research: Open people prefer complex, unconventional, diverse music
        openness = (
            features['genre_diversity'] * 0.25 +
            features['artist_diversity'] * 0.15 +
            features['mainstream_avoidance'] * 0.15 +
            features['instrumentalness'] * 0.10 +
            features['musical_complexity'] * 0.10 +
            features['key_diversity'] * 0.08 +
            features['time_signature_diversity'] * 0.07 +
            features['unpopular_preference'] * 0.10
        )
        predictions['Openness'] = np.clip(openness * 5, 1, 5)
        
        # CONSCIENTIOUSNESS - Organized, disciplined, conventional
        # Research: Conscientious people prefer mainstream, predictable music
        conscientiousness = (
            features['mainstream_preference'] * 0.30 +
            (1 - features['emotional_variance']) * 0.20 +
            (1 - features['energy_variance']) * 0.15 +
            features['popularity'] / 100 * 0.15 +
            (1 - features['genre_diversity']) * 0.10 +
            features['danceability'] * 0.10
        )
        predictions['Conscientiousness'] = np.clip(conscientiousness * 5, 1, 5)
        
        # AGREEABLENESS - Cooperative, trusting, helpful
        # Research: Agreeable people prefer positive, harmonious, conventional music
        agreeableness = (
            features['valence'] * 0.30 +
            features['positive_music_ratio'] * 0.20 +
            (1 - features['negative_music_ratio']) * 0.15 +
            features['mainstream_preference'] * 0.15 +
            features['acoustic_ratio'] * 0.10 +
            (1 - features['loudness'] / -60) * 0.10  # Prefer quieter music
        )
        predictions['Agreeableness'] = np.clip(agreeableness * 5, 1, 5)
        
        # NEUROTICISM - Emotional instability, anxiety, moodiness
        # Research: Neurotic people prefer sad, emotional, complex music
        neuroticism = (
            (1 - features['valence']) * 0.25 +
            features['emotional_variance'] * 0.20 +
            features['negative_music_ratio'] * 0.15 +
            features['energy_variance'] * 0.15 +
            (1 - features['positive_music_ratio']) * 0.10 +
            features['acousticness'] * 0.10 +
            (1 - features['danceability']) * 0.05
        )
        predictions['Neuroticism'] = np.clip(neuroticism * 5, 1, 5)
        
        # Round to 2 decimal places
        for trait in predictions:
            predictions[trait] = round(predictions[trait], 2)
        
        return predictions

def create_personality_insights(predictions):
    """Generate detailed personality insights"""
    
    insights = {}
    
    descriptions = {
        'Extraversion': {
            'high': "🎉 **Social Music Lover!** You gravitate toward energetic, danceable music that gets people moving. Your playlists are perfect for parties and social gatherings. You likely enjoy mainstream hits and music that creates a fun, social atmosphere.",
            'medium': "🎵 **Balanced Social Energy** - You enjoy both upbeat social music and quieter personal listening. Your taste adapts well to different social situations.",
            'low': "🎧 **Introspective Listener** - You prefer quieter, more contemplative music perfect for solo listening and reflection. You value music as personal experience over social sharing."
        },
        'Openness': {
            'high': "🎨 **Musical Explorer!** You're always discovering new artists, genres, and experimental sounds. You love unusual music that others might not 'get' and take pride in your unique taste. You probably have incredibly diverse playlists.",
            'medium': "🎼 **Curious but Grounded** - You balance musical exploration with familiar favorites. You enjoy discovering new music but also appreciate established artists.",
            'low': "📻 **Reliable Favorites** - You know what you like and stick with it! You prefer familiar genres and artists that consistently deliver what you enjoy. There's comfort in musical predictability."
        },
        'Conscientiousness': {
            'high': "📋 **Organized Music Habits** - Your music listening is structured and consistent. You probably have well-organized playlists, stick to routines, and prefer mainstream music that 'makes sense.' You like musical reliability.",
            'medium': "⚖️ **Structured but Flexible** - You have some organization in your music habits but can be spontaneous when the mood strikes.",
            'low': "🎲 **Spontaneous Music Spirit** - Your musical choices are driven by mood and moment! You're not tied to routines and enjoy letting your musical taste wander wherever it wants to go."
        },
        'Agreeableness': {
            'high': "🤝 **Harmony Seeker** - You love music that brings people together! You prefer positive, uplifting songs that create good vibes. You probably enjoy music that most people can appreciate and sing along to.",
            'medium': "🎶 **Emotionally Balanced** - You appreciate both uplifting music and more complex emotional expressions, adapting to different social contexts.",
            'low': "🎸 **Edge Appreciator** - You're drawn to more intense, unconventional, or emotionally complex music. You don't need everyone to 'get' your music taste - you value authenticity over popularity."
        },
        'Neuroticism': {
            'high': "💭 **Emotional Music Connection** - Music is your emotional outlet! You're drawn to songs that match and help you process complex feelings. You might use music therapeutically and appreciate artists who aren't afraid of emotional depth.",
            'medium': "🌊 **Mood-Responsive Listening** - Your music choices reflect your emotional state, ranging from comforting to energizing depending on what you need.",
            'low': "☀️ **Stable Mood Music** - You prefer music that maintains positive vibes and emotional balance. You use music to stay upbeat rather than to process difficult emotions."
        }
    }
    
    for trait, score in predictions.items():
        if score >= 3.5:
            category = 'high'
            level = 'High'
        elif score <= 2.5:
            category = 'low' 
            level = 'Low'
        else:
            category = 'medium'
            level = 'Moderate'
        
        insights[trait] = {
            'score': score,
            'level': level,
            'description': descriptions[trait][category]
        }
    
    return insights

def main():
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits based on your Spotify listening habits!")
    
    # Check if we have the auth code in URL
    query_params = st.experimental_get_query_params()
    
    # Initialize the app
    app = SpotifyPersonalityApp()
    
    # Handle authentication flow
    if 'code' in query_params:
        # User has returned from Spotify auth
        auth_code = query_params['code'][0]
        
        with st.spinner("🔐 Authenticating with Spotify..."):
            if app.authenticate_user(auth_code):
                st.success("✅ Successfully connected to your Spotify account!")
                
                # Get and analyze music data
                with st.spinner("🎵 Analyzing your music library... This may take a moment..."):
                    music_data = app.get_user_music_data()
                
                if music_data:
                    # Extract features and predict personality
                    features = app.extract_personality_features(music_data)
                    predictions = app.predict_personality(features)
                    insights = create_personality_insights(predictions)
                    
                    # Display results
                    st.header("🎯 Your Musical Personality Profile")
                    
                    # Radar chart
                    fig = go.Figure()
                    
                    traits = list(predictions.keys())
                    scores = list(predictions.values())
                    
                    fig.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=traits,
                        fill='toself',
                        name='Your Personality',
                        line_color='rgb(34, 139, 34)',
                        fillcolor='rgba(34, 139, 34, 0.3)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[1, 5],
                                tickmode='linear',
                                tick0=1,
                                dtick=1
                            )),
                        showlegend=False,
                        title={
                            'text': "Your Big Five Personality Traits (1-5 scale)",
                            'x': 0.5,
                            'font': {'size': 20}
                        },
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Personality breakdown
                    st.header("📝 Your Personality Analysis")
                    
                    for trait, insight in insights.items():
                        with st.expander(f"**{trait}**: {insight['score']:.1f}/5.0 - {insight['level']}"):
                            st.markdown(insight['description'])
                            st.progress(insight['score'] / 5.0)
                    
                    # Music statistics
                    st.header("🎼 Your Music Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎵 Tracks Analyzed", len(music_data['audio_features']))
                        st.metric("⚡ Avg Energy", f"{features['energy']:.2f}")
                        st.metric("💃 Avg Danceability", f"{features['danceability']:.2f}")
                    
                    with col2:
                        st.metric("🌍 Unique Genres", features['unique_genres_count'])
                        st.metric("😊 Avg Positivity", f"{features['valence']:.2f}")
                        st.metric("🎸 Acousticness", f"{features['acousticness']:.2f}")
                    
                    with col3:
                        st.metric("📈 Avg Popularity", f"{features['popularity']:.0f}/100")
                        st.metric("🎼 Instrumentalness", f"{features['instrumentalness']:.2f}")
                        st.metric("🎤 Speechiness", f"{features['speechiness']:.2f}")
                    
                    with col4:
                        st.metric("🎹 Key Diversity", f"{features['key_diversity']:.2f}")
                        st.metric("🔄 Genre Diversity", f"{features['genre_diversity']:.2f}")
                        st.metric("🎯 Mainstream %", f"{features['mainstream_preference']:.0%}")
                    
                    # Genre breakdown
                    if music_data['genres']:
                        st.header("🎵 Your Music Genres")
                        genre_counts = Counter(music_data['genres'])
                        top_genres = dict(genre_counts.most_common(10))
                        
                        genre_df = pd.DataFrame(list(top_genres.items()), columns=['Genre', 'Count'])
                        st.bar_chart(genre_df.set_index('Genre'))
                    
                    # Try again button
                    if st.button("🔄 Analyze Again"):
                        st.experimental_rerun()
                
            else:
                st.error("❌ Authentication failed. Please try again.")
                if st.button("🔄 Try Again"):
                    st.experimental_rerun()
    
    else:
        # Show login page
        st.markdown("""
        ### How it works:
        1. 🔐 **Connect your Spotify** - We'll analyze your listening history
        2. 🎵 **We analyze your music** - Looking at audio features, genres, and listening patterns  
        3. 🧠 **Get your personality profile** - Based on psychological research about music and personality
        4. 📊 **See detailed insights** - Learn what your music taste reveals about you!
        
        ### What we analyze:
        - 🎼 **Audio features** (energy, danceability, positivity, etc.)
        - 🌍 **Genre diversity** and exploration patterns
        - 📈 **Popularity preferences** (mainstream vs. niche)
        - 🎯 **Listening consistency** and habits
        
        **Ready to discover your musical personality?**
        """)
        
        # Get auth URL and display login button
        auth_url = app.get_auth_url()
        
        st.markdown(f"""
        <div style="text-align: center; margin: 30px 0;">
            <a href="{auth_url}" target="_self">
                <button style="
                    background-color: #1DB954;
                    color: white;
                    padding: 15px 30px;
                    font-size: 18px;
                    border: none;
                    border-radius: 50px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                ">
                    🎵 Connect with Spotify
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ---
        
        **Privacy Note:** We only read your music data to generate your personality profile. 
        We don't store your data or access your private information beyond what's needed for the analysis.
        """)

if __name__ == "__main__":
    main()