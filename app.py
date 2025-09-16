# app.py - Complete Spotify Personality Predictor
import os
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

# Use the EXACT same auth pattern as your working code
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

# Page config
st.set_page_config(
    page_title="Spotify Personality Predictor",
    page_icon="🎵",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

# -----------------------------------------------------------------------------
# Auth - EXACT SAME AS YOUR WORKING CODE
# -----------------------------------------------------------------------------
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
        # Clear the code from URL
        st.query_params.clear()
        st.rerun()

    login_url = auth_manager.get_authorize_url()
    st.info("Please log in with Spotify")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

# -----------------------------------------------------------------------------
# Music Data Collection
# -----------------------------------------------------------------------------
def get_user_music_data(sp, limit=50):
    """Get user's music data with audio features"""
    try:
        # Get user's top tracks from different time periods
        st.write("🎵 Fetching your top tracks...")
        top_tracks_data = {}
        
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                tracks = sp.current_user_top_tracks(limit=limit, time_range=time_range)
                top_tracks_data[time_range] = tracks['items']
                st.write(f"✅ Got {len(tracks['items'])} tracks from {time_range.replace('_', ' ')}")
            except Exception as e:
                st.warning(f"Could not get {time_range} tracks: {e}")
                top_tracks_data[time_range] = []
        
        # Get recent tracks
        st.write("🎵 Fetching your recent listening history...")
        try:
            recent_tracks = sp.current_user_recently_played(limit=limit)
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
        
        # Get audio features - THE KEY PART
        st.write("🎵 Getting audio features...")
        track_ids = [track['id'] for track in unique_tracks if track['id']]
        
        # Spotify audio_features() can handle up to 100 tracks per request
        all_audio_features = []
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i+100]
            st.write(f"   Processing batch {i//100 + 1}/{(len(track_ids)-1)//100 + 1}...")
            
            try:
                batch_features = sp.audio_features(batch_ids)
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
            return None
        
        st.success(f"✅ Successfully got audio features for {len(all_audio_features)} tracks!")
        
        # Get artists for genre analysis
        st.write("🎤 Getting artist information...")
        artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks if track['artists']]))
        
        all_artists = []
        for i in range(0, len(artist_ids), 50):  # Max 50 artists per request
            batch_ids = artist_ids[i:i+50]
            try:
                artists_response = sp.artists(batch_ids)
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

# -----------------------------------------------------------------------------
# Personality Analysis Functions
# -----------------------------------------------------------------------------
def extract_personality_features(music_data):
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

def predict_personality(features):
    """Predict Big Five personality traits using research-backed approach"""
    
    predictions = {}
    
    # EXTRAVERSION - Social, energetic, outgoing
    extraversion = (
        features['energy'] * 0.25 +
        features['danceability'] * 0.20 +
        features['valence'] * 0.15 +
        features['high_energy_ratio'] * 0.15 +
        features['danceable_ratio'] * 0.10 +
        (features['loudness'] + 60) / 60 * 0.10 +  # Normalize loudness
        features['mainstream_preference'] * 0.05
    )
    predictions['Extraversion'] = np.clip(extraversion * 5, 1, 5)
    
    # OPENNESS - Creative, curious, open to new experiences
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
            'high': "🎉 **Social Music Lover!** You gravitate toward energetic, danceable music that gets people moving. Your playlists are perfect for parties and social gatherings.",
            'medium': "🎵 **Balanced Social Energy** - You enjoy both upbeat social music and quieter personal listening.",
            'low': "🎧 **Introspective Listener** - You prefer quieter, more contemplative music perfect for solo listening and reflection."
        },
        'Openness': {
            'high': "🎨 **Musical Explorer!** You're always discovering new artists, genres, and experimental sounds. You love unusual music that others might not 'get'.",
            'medium': "🎼 **Curious but Grounded** - You balance musical exploration with familiar favorites.",
            'low': "📻 **Reliable Favorites** - You know what you like and stick with it! You prefer familiar genres and artists."
        },
        'Conscientiousness': {
            'high': "📋 **Organized Music Habits** - Your music listening is structured and consistent. You probably have well-organized playlists.",
            'medium': "⚖️ **Structured but Flexible** - You have some organization in your music habits but can be spontaneous.",
            'low': "🎲 **Spontaneous Music Spirit** - Your musical choices are driven by mood and moment!"
        },
        'Agreeableness': {
            'high': "🤝 **Harmony Seeker** - You love music that brings people together! You prefer positive, uplifting songs that create good vibes.",
            'medium': "🎶 **Emotionally Balanced** - You appreciate both uplifting music and more complex emotional expressions.",
            'low': "🎸 **Edge Appreciator** - You're drawn to more intense, unconventional, or emotionally complex music."
        },
        'Neuroticism': {
            'high': "💭 **Emotional Music Connection** - Music is your emotional outlet! You're drawn to songs that help you process complex feelings.",
            'medium': "🌊 **Mood-Responsive Listening** - Your music choices reflect your emotional state.",
            'low': "☀️ **Stable Mood Music** - You prefer music that maintains positive vibes and emotional balance."
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

# -----------------------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------------------
def main():
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("### Discover your Big Five personality traits based on your Spotify listening habits!")
    
    # Use the EXACT same auth pattern as your working code
    sp = ensure_spotify_client()
    st.success("Spotify authenticated ✅")

    # Introduction
    st.markdown("""
    ### How it works:
    1. 🔐 **We analyze your Spotify data** - Your top tracks, recent plays, and music features
    2. 🎵 **Extract musical patterns** - Energy, genres, popularity, emotional content, etc.
    3. 🧠 **Predict personality traits** - Based on psychological research linking music to personality
    4. 📊 **Show your results** - Detailed breakdown of your Big Five personality profile
    """)

    if st.button("🧠 Analyze My Musical Personality", type="primary"):
        with st.spinner("🎵 Analyzing your music library... This may take a moment..."):
            music_data = get_user_music_data(sp)
        
        if music_data:
            # Extract features and predict personality
            with st.spinner("🧠 Calculating your personality traits..."):
                features = extract_personality_features(music_data)
                predictions = predict_personality(features)
                insights = create_personality_insights(predictions)
            
            # Display results
            st.header("🎯 Your Musical Personality Profile")
            
            # Summary of top traits
            sorted_traits = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            top_trait = sorted_traits[0]
            st.info(f"🌟 **Your dominant trait is {top_trait[0]}** with a score of {top_trait[1]:.1f}/5.0")
            
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
                st.header("🎵 Your Top Music Genres")
                genre_counts = Counter(music_data['genres'])
                top_genres = dict(genre_counts.most_common(15))
                
                if top_genres:
                    genre_df = pd.DataFrame(list(top_genres.items()), columns=['Genre', 'Count'])
                    st.bar_chart(genre_df.set_index('Genre'))
            
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
            
            if features['instrumental_ratio'] > 0.3:
                facts.append("🎼 You appreciate instrumental music more than most!")
            
            if features['acoustic_ratio'] > 0.5:
                facts.append("🎸 You have a strong preference for acoustic sounds!")
            
            for fact in facts:
                st.info(fact)
            
            # Try again button
            st.markdown("---")
            if st.button("🔄 Analyze Again"):
                st.rerun()
        
        else:
            st.error("Could not analyze your music. Please make sure you have some listening history on Spotify!")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Privacy Note:** We only read your music data to generate your personality profile. 
    We don't store your data or access your private information beyond what's needed for the analysis.
    
    **Disclaimer:** This is for entertainment purposes. Personality predictions are based on research correlations 
    but should not be considered definitive psychological assessments.
    """)

if __name__ == "__main__":
    main()