# app.py - Deployment ready Spotify personality app
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import os
import urllib.parse

# Page config
st.set_page_config(
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

class SpotifyPersonalityApp:
    def __init__(self, client_id, client_secret, redirect_uri):
        # Comprehensive scope for better analysis
        scope = "user-read-recently-played user-top-read user-read-private playlist-read-private user-library-read"
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            cache_path=None,  # Don't cache in deployment
            show_dialog=True
        ))
    
    def get_user_music_features(self, limit=50):
        """Extract meaningful features from user's Spotify data"""
        try:
            # Get comprehensive user data
            user_data = {}
            
            # Get top tracks from different time ranges
            user_data['short_term'] = self.sp.current_user_top_tracks(limit=limit, time_range='short_term')
            user_data['medium_term'] = self.sp.current_user_top_tracks(limit=limit, time_range='medium_term')
            user_data['long_term'] = self.sp.current_user_top_tracks(limit=limit, time_range='long_term')
            
            # Get recent tracks
            user_data['recent'] = self.sp.current_user_recently_played(limit=limit)
            
            # Collect all track IDs
            all_track_ids = []
            all_tracks = []
            
            # From top tracks
            for time_range in ['short_term', 'medium_term', 'long_term']:
                for track in user_data[time_range]['items']:
                    all_track_ids.append(track['id'])
                    all_tracks.append(track)
            
            # From recent tracks
            for item in user_data['recent']['items']:
                track = item['track']
                all_track_ids.append(track['id'])
                all_tracks.append(track)
            
            if not all_track_ids:
                return None
            
            # Remove duplicates while preserving order
            unique_track_ids = list(dict.fromkeys(all_track_ids))
            
            # Get audio features (Spotify limits to 100 per request)
            audio_features = []
            for i in range(0, len(unique_track_ids), 100):
                batch = unique_track_ids[i:i+100]
                features = self.sp.audio_features(batch)
                audio_features.extend([f for f in features if f is not None])
            
            if not audio_features:
                return None
            
            # Get artist genres for diversity calculation
            artist_ids = list(set([track['artists'][0]['id'] for track in all_tracks]))
            artists_data = []
            
            # Get artists in batches of 50 (Spotify limit)
            for i in range(0, len(artist_ids), 50):
                batch = artist_ids[i:i+50]
                artists = self.sp.artists(batch)
                artists_data.extend(artists['artists'])
            
            # Calculate genre diversity
            all_genres = []
            for artist in artists_data:
                all_genres.extend(artist['genres'])
            
            genre_diversity = len(set(all_genres)) / max(len(all_genres), 1)
            unique_genres = len(set(all_genres))
            
            # Calculate artist diversity  
            unique_artists = len(set([track['artists'][0]['id'] for track in all_tracks]))
            artist_diversity = unique_artists / max(len(all_tracks), 1)
            
            # Process audio features
            features_df = pd.DataFrame(audio_features)
            
            # Calculate behavioral features
            user_features = {
                # Basic audio features
                'energy': features_df['energy'].mean(),
                'danceability': features_df['danceability'].mean(),
                'valence': features_df['valence'].mean(),
                'acousticness': features_df['acousticness'].mean(),
                'instrumentalness': features_df['instrumentalness'].mean(),
                'speechiness': features_df['speechiness'].mean(),
                'loudness': features_df['loudness'].mean(),
                'tempo': features_df['tempo'].mean(),
                'liveness': features_df['liveness'].mean(),
                
                # Popularity and mainstream preference
                'popularity': features_df['popularity'].mean(),
                'mainstream_preference': (features_df['popularity'] > 70).mean(),
                
                # Diversity metrics
                'genre_diversity': genre_diversity,
                'unique_genres': unique_genres,
                'artist_diversity': artist_diversity,
                
                # Emotional patterns
                'emotional_variance': features_df['valence'].std(),
                'energy_variance': features_df['energy'].std(),
                'positive_music_ratio': (features_df['valence'] > 0.6).mean(),
                'high_energy_ratio': (features_df['energy'] > 0.7).mean(),
                'danceable_ratio': (features_df['danceability'] > 0.7).mean(),
                
                # Complexity indicators
                'instrumental_ratio': (features_df['instrumentalness'] > 0.5).mean(),
                'acoustic_ratio': (features_df['acousticness'] > 0.5).mean(),
                'live_music_ratio': (features_df['liveness'] > 0.8).mean(),
                
                # Musical characteristics
                'tempo_variance': features_df['tempo'].std(),
                'key_diversity': len(features_df['key'].unique()) / 12,
                'mode_diversity': len(features_df['mode'].unique()) / 2,
                
                # Time-based consistency (simplified)
                'listening_consistency': 0.7,  # Would calculate from actual listening patterns
            }
            
            return user_features, features_df, all_genres[:20]  # Return sample genres for display
            
        except Exception as e:
            st.error(f"Error fetching user data: {e}")
            return None, None, None
    
    def predict_personality_research_based(self, features):
        """Predict personality using research-backed rules"""
        
        # Research-backed personality prediction rules
        personality_scores = {}
        
        # EXTRAVERSION - energetic, social, outgoing music
        extraversion_score = (
            features['energy'] * 0.25 +
            features['danceability'] * 0.25 +
            (features['loudness'] + 60) / 60 * 0.15 +  # Normalize loudness
            features['valence'] * 0.15 +
            features['high_energy_ratio'] * 0.10 +
            features['danceable_ratio'] * 0.10
        )
        personality_scores['Extraversion'] = np.clip(extraversion_score * 5, 1, 5)
        
        # OPENNESS - diverse, complex, experimental music
        openness_score = (
            features['genre_diversity'] * 0.20 +
            features['artist_diversity'] * 0.15 +
            features['instrumentalness'] * 0.15 +
            (1 - features['mainstream_preference']) * 0.15 +
            features['acousticness'] * 0.10 +
            features['speechiness'] * 0.10 +
            features['key_diversity'] * 0.10 +
            (1 - features['popularity'] / 100) * 0.05  # Less popular = more open
        )
        personality_scores['Openness'] = np.clip(openness_score * 5, 1, 5)
        
        # CONSCIENTIOUSNESS - consistent, mainstream, organized
        conscientiousness_score = (
            features['mainstream_preference'] * 0.25 +
            features['listening_consistency'] * 0.20 +
            (features['popularity'] / 100) * 0.15 +
            (1 - features['emotional_variance']) * 0.15 +
            (1 - features['energy_variance']) * 0.10 +
            features['danceability'] * 0.10 +
            (1 - features['genre_diversity']) * 0.05  # Less diversity = more conscientious
        )
        personality_scores['Conscientiousness'] = np.clip(conscientiousness_score * 5, 1, 5)
        
        # AGREEABLENESS - positive, harmonious, popular music
        agreeableness_score = (
            features['valence'] * 0.30 +
            features['positive_music_ratio'] * 0.20 +
            features['mainstream_preference'] * 0.15 +
            (1 - features['loudness'] / -60) * 0.10 +  # Quieter music
            features['danceability'] * 0.10 +
            (1 - features['energy_variance']) * 0.10 +
            features['acoustic_ratio'] * 0.05
        )
        personality_scores['Agreeableness'] = np.clip(agreeableness_score * 5, 1, 5)
        
        # NEUROTICISM - emotional, variable, less positive music
        neuroticism_score = (
            (1 - features['valence']) * 0.25 +
            features['emotional_variance'] * 0.20 +
            features['energy_variance'] * 0.15 +
            (1 - features['positive_music_ratio']) * 0.15 +
            features['acousticness'] * 0.10 +
            (1 - features['danceability']) * 0.10 +
            features['speechiness'] * 0.05
        )
        personality_scores['Neuroticism'] = np.clip(neuroticism_score * 5, 1, 5)
        
        # Round to 2 decimal places
        for trait in personality_scores:
            personality_scores[trait] = round(personality_scores[trait], 2)
        
        return personality_scores

def get_deployment_redirect_uri():
    """Get the correct redirect URI for current deployment"""
    
    # Check if running on Streamlit Cloud
    if 'STREAMLIT_SERVER_PORT' in os.environ:
        # Get the current URL from Streamlit
        try:
            # This works for Streamlit Cloud
            return f"https://{st.experimental_get_query_params().get('streamlit_url', [''])[0]}"
        except:
            pass
    
    # Fallback options
    return "http://localhost:8501/"  # Default Streamlit port

def create_personality_insights(predictions):
    """Generate personality insights and descriptions"""
    
    insights = {}
    
    trait_descriptions = {
        'Extraversion': {
            'high': "🎉 You love energetic, social music! Your playlist probably gets the party started. You're drawn to upbeat, danceable tracks that match your outgoing personality.",
            'medium': "🎵 You enjoy a good mix of energetic and chill music, adapting to different social situations and moods.",
            'low': "🎧 You prefer quieter, more introspective music. Perfect for solo listening and deep thinking sessions."
        },
        'Openness': {
            'high': "🎨 You're a musical explorer! You love discovering new artists, genres, and experimental sounds. Your taste is unique and adventurous.",
            'medium': "🎼 You balance familiar favorites with new discoveries, enjoying both mainstream hits and hidden gems.",
            'low': "📻 You know what you like! You stick to familiar genres and artists that consistently deliver what you enjoy."
        },
        'Conscientiousness': {
            'high': "📋 Your music habits are well-organized! You probably have carefully curated playlists and consistent listening routines.",
            'medium': "⚖️ You balance structure with spontaneity in your music choices.",
            'low': "🎲 Your music taste is spontaneous and varied - you follow your mood wherever it takes you!"
        },
        'Agreeableness': {
            'high': "🤝 You love music that brings people together! You prefer positive, harmonious songs that create good vibes for everyone.",
            'medium': "🎶 You enjoy both uplifting music and more complex emotional expressions.",
            'low': "🎸 You're drawn to more intense or unconventional music that others might find challenging."
        },
        'Neuroticism': {
            'high': "💭 Your music reflects emotional depth and complexity. You may use music to process feelings and find comfort during difficult times.",
            'medium': "🌊 You enjoy both emotionally rich and stable, calming music depending on your state of mind.",
            'low': "☀️ You prefer stable, positive music that maintains good vibes and emotional balance."
        }
    }
    
    for trait, score in predictions.items():
        if score >= 3.5:
            category = 'high'
        elif score <= 2.5:
            category = 'low'
        else:
            category = 'medium'
        
        insights[trait] = {
            'score': score,
            'category': category,
            'description': trait_descriptions[trait][category]
        }
    
    return insights

def main():
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits based on your Spotify listening habits!")
    
    # Instructions for getting Spotify credentials
    with st.expander("🔑 How to get Spotify API credentials", expanded=False):
        st.markdown("""
        1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
        2. Log in with your Spotify account
        3. Click **"Create App"**
        4. Fill in:
           - **App name**: "My Personality App"
           - **App description**: "Personality prediction from music"
           - **Redirect URI**: `https://your-app-name.streamlit.app/` (replace with your actual URL)
        5. Copy your **Client ID** and **Client Secret**
        6. Come back here and enter them below!
        """)
    
    # Sidebar for credentials
    st.sidebar.title("🎵 Spotify Setup")
    
    # Get current app URL for redirect URI hint
    try:
        current_url = st.experimental_get_query_params()
        if current_url:
            suggested_redirect = "https://your-app-name.streamlit.app/"
        else:
            suggested_redirect = "http://localhost:8501/"
    except:
        suggested_redirect = "https://your-app-name.streamlit.app/"
    
    st.sidebar.info(f"💡 Your redirect URI should be:\n`{suggested_redirect}`")
    
    # Try to get credentials from Streamlit secrets first
    try:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
        st.sidebar.success("✅ Using credentials from Streamlit secrets")
        
        # Get redirect URI from secrets or use default
        try:
            redirect_uri = st.secrets["SPOTIFY_REDIRECT_URI"]
        except KeyError:
            redirect_uri = suggested_redirect
            st.sidebar.info(f"Using default redirect URI: {redirect_uri}")
        
    except KeyError:
        # Fallback to manual input if secrets not available
        st.sidebar.warning("⚠️ Spotify credentials not found in secrets")
        
        # Input fields
        client_id = st.sidebar.text_input("Spotify Client ID", type="password", help="From your Spotify app dashboard")
        client_secret = st.sidebar.text_input("Spotify Client Secret", type="password", help="From your Spotify app dashboard")
        
        # Allow custom redirect URI
        redirect_uri = st.sidebar.text_input("Redirect URI", value=suggested_redirect, help="Must match your Spotify app settings")
        
        if not client_id or not client_secret:
            st.warning("👆 Please enter your Spotify API credentials in the sidebar to get started!")
            st.info("Don't have Spotify credentials? Click the 🔑 section above to learn how to get them!")
            return
    
    # Initialize app
    try:
        app = SpotifyPersonalityApp(client_id, client_secret, redirect_uri)
        
        st.sidebar.success("✅ Spotify credentials loaded!")
        
        # Main analysis button
        if st.button("🎯 Analyze My Music Personality", type="primary"):
            
            with st.spinner("🎵 Fetching your Spotify data..."):
                try:
                    result = app.get_user_music_features()
                    
                    if result is None:
                        st.error("❌ Could not fetch your Spotify data. Please check your credentials and try again.")
                        return
                    
                    features, audio_features_df, sample_genres = result
                    
                    st.success(f"✅ Analyzed {len(audio_features_df)} tracks from your library!")
                    
                except Exception as e:
                    st.error(f"❌ Error fetching data: {e}")
                    st.info("💡 Make sure you have some listening history in Spotify and try again.")
                    return
            
            with st.spinner("🧠 Analyzing your personality..."):
                # Predict personality
                predictions = app.predict_personality_research_based(features)
                insights = create_personality_insights(predictions)
            
            # Display results
            st.header("🎯 Your Musical Personality Profile")
            
            # Create radar chart
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
                    'text': "Your Big Five Personality Traits",
                    'x': 0.5,
                    'font': {'size': 20}
                },
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display personality scores and insights
            st.header("📝 Your Personality Breakdown")
            
            for trait, insight in insights.items():
                with st.expander(f"**{trait}**: {insight['score']}/5.0 ({insight['category'].title()})"):
                    st.write(insight['description'])
                    
                    # Progress bar for visual representation
                    progress = insight['score'] / 5.0
                    st.progress(progress)
            
            # Music characteristics
            st.header("🎼 Your Music DNA")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎵 Energy Level", f"{features['energy']:.2f}", help="How energetic your music is (0-1)")
                st.metric("💃 Danceability", f"{features['danceability']:.2f}", help="How danceable your music is (0-1)")
                st.metric("😊 Positivity", f"{features['valence']:.2f}", help="How positive/happy your music is (0-1)")
            
            with col2:
                st.metric("🎸 Acousticness", f"{features['acousticness']:.2f}", help="How acoustic vs electric your music is (0-1)")
                st.metric("🎼 Instrumentalness", f"{features['instrumentalness']:.2f}", help="How much instrumental music you like (0-1)")
                st.metric("📈 Popularity", f"{features['popularity']:.0f}/100", help="How mainstream your music taste is")
            
            with col3:
                st.metric("🌍 Genre Diversity", f"{features['unique_genres']}", help="Number of unique genres in your music")
                st.metric("🎨 Artist Variety", f"{features['artist_diversity']:.2f}", help="How diverse your artist choices are (0-1)")
                st.metric("🎯 Mainstream Score", f"{features['mainstream_preference']:.2f}", help="How much you prefer popular music (0-1)")
            
            # Show some genres
            if sample_genres:
                st.header("🎵 Your Music Genres")
                genre_text = ", ".join(sample_genres[:15])
                st.write(f"**Sample genres from your music:** {genre_text}")
            
            # Fun facts
            st.header("🎉 Fun Facts About Your Music")
            facts = []
            
            if features['energy'] > 0.7:
                facts.append("⚡ You love high-energy music that gets your blood pumping!")
            elif features['energy'] < 0.3:
                facts.append("🌙 You prefer calm, low-energy music for relaxation.")
            
            if features['danceability'] > 0.7:
                facts.append("💃 Your music could definitely get a party started!")
            
            if features['valence'] > 0.7:
                facts.append("😊 You're drawn to positive, uplifting music!")
            elif features['valence'] < 0.3:
                facts.append("🎭 You appreciate more melancholic or complex emotional music.")
            
            if features['genre_diversity'] > 0.5:
                facts.append("🌈 You're a musical explorer with diverse taste!")
            
            if features['mainstream_preference'] > 0.7:
                facts.append("📻 You know what's popular and you like it!")
            elif features['mainstream_preference'] < 0.3:
                facts.append("🎧 You're a musical hipster who avoids the mainstream!")
            
            for fact in facts:
                st.info(fact)
            
            # Share results
            st.header("📱 Share Your Results")
            share_text = f"I just discovered my musical personality! My top traits: {max(predictions.items(), key=lambda x: x[1])[0]} ({max(predictions.values()):.1f}/5.0). Check yours: [Your App URL]"
            st.code(share_text, language=None)
    
    except Exception as e:
        st.error(f"❌ Error initializing Spotify connection: {e}")
        st.info("💡 Double-check your Client ID, Client Secret, and make sure your Redirect URI matches exactly what you set in your Spotify app settings.")

if __name__ == "__main__":
    main()