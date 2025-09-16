# spotify_personality_app.py - Complete implementation for your app

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from collections import Counter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class SpotifyPersonalityApp:
    def __init__(self, client_id, client_secret, redirect_uri):
        """Initialize Spotify API connection"""
        scope = "user-read-recently-played user-top-read playlist-read-private user-library-read"
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        ))
        
        # Load trained models
        self.models = self._load_models()
    
    def _load_models(self):
        """Load pre-trained personality prediction models"""
        try:
            return joblib.load("models/behavioral_personality_models.pkl")
        except FileNotFoundError:
            st.error("Model file not found. Please train the model first.")
            return None
    
    def get_comprehensive_user_data(self, limit=50):
        """Get comprehensive user data from Spotify"""
        try:
            # Get different types of data
            data = {
                'recent_tracks': self.sp.current_user_recently_played(limit=limit),
                'top_tracks_short': self.sp.current_user_top_tracks(time_range='short_term', limit=limit),
                'top_tracks_medium': self.sp.current_user_top_tracks(time_range='medium_term', limit=limit),
                'top_tracks_long': self.sp.current_user_top_tracks(time_range='long_term', limit=limit),
                'top_artists_short': self.sp.current_user_top_artists(time_range='short_term', limit=20),
                'top_artists_medium': self.sp.current_user_top_artists(time_range='medium_term', limit=20),
                'top_artists_long': self.sp.current_user_top_artists(time_range='long_term', limit=20),
                'saved_tracks': self.sp.current_user_saved_tracks(limit=limit),
                'playlists': self.sp.current_user_playlists(limit=20)
            }
            
            # Get audio features for all tracks
            all_track_ids = []
            
            # Collect track IDs from all sources
            for tracks in [data['recent_tracks']['items'], 
                          data['top_tracks_short']['items'],
                          data['top_tracks_medium']['items'],
                          data['top_tracks_long']['items']]:
                for item in tracks:
                    track_id = item['track']['id'] if 'track' in item else item['id']
                    if track_id:
                        all_track_ids.append(track_id)
            
            # Add saved tracks
            for item in data['saved_tracks']['items']:
                if item['track']['id']:
                    all_track_ids.append(item['track']['id'])
            
            # Remove duplicates and get audio features
            unique_track_ids = list(set(all_track_ids))
            
            # Spotify API limits batch requests to 100
            audio_features = []
            for i in range(0, len(unique_track_ids), 100):
                batch = unique_track_ids[i:i+100]
                features = self.sp.audio_features(batch)
                audio_features.extend([f for f in features if f is not None])
            
            data['audio_features'] = audio_features
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching user data: {e}")
            return None
    
    def extract_behavioral_features(self, user_data):
        """Extract behavioral features from Spotify data"""
        if not user_data or not user_data.get('audio_features'):
            return None
        
        # Convert audio features to DataFrame
        audio_df = pd.DataFrame(user_data['audio_features'])
        
        if audio_df.empty:
            return None
        
        features = {}
        
        # === EXTRAVERSION FEATURES ===
        features['avg_energy'] = audio_df['energy'].mean()
        features['avg_danceability'] = audio_df['danceability'].mean()
        features['avg_loudness'] = audio_df['loudness'].mean()
        features['avg_tempo'] = audio_df['tempo'].mean()
        features['high_energy_ratio'] = (audio_df['energy'] > 0.7).mean()
        features['danceable_ratio'] = (audio_df['danceability'] > 0.7).mean()
        
        # Social music indicator (high valence + danceability)
        features['social_music_score'] = (audio_df['valence'] + audio_df['danceability']).mean() / 2
        
        # === OPENNESS FEATURES ===
        features['avg_instrumentalness'] = audio_df['instrumentalness'].mean()
        features['avg_acousticness'] = audio_df['acousticness'].mean()
        features['tempo_variance'] = audio_df['tempo'].std()
        features['key_diversity'] = len(audio_df['key'].unique()) / 12
        features['mode_diversity'] = len(audio_df['mode'].unique()) / 2
        features['time_signature_diversity'] = len(audio_df['time_signature'].unique()) / 7
        features['instrumental_ratio'] = (audio_df['instrumentalness'] > 0.5).mean()
        
        # Musical complexity
        features['musical_complexity'] = (
            audio_df['instrumentalness'] + 
            audio_df['acousticness'] + 
            (1 - audio_df['popularity'] / 100)  # Less popular = more complex/experimental
        ).mean() / 3
        
        # === CONSCIENTIOUSNESS FEATURES ===
        # Consistency in preferences (low variance = more consistent)
        features['valence_consistency'] = 1 / (1 + audio_df['valence'].std())
        features['energy_consistency'] = 1 / (1 + audio_df['energy'].std())
        features['tempo_consistency'] = 1 / (1 + audio_df['tempo'].std() / 100)
        
        # Preference for familiar/popular music
        features['mainstream_preference'] = audio_df['popularity'].mean() / 100
        
        # === AGREEABLENESS FEATURES ===
        features['avg_valence'] = audio_df['valence'].mean()
        features['positive_music_ratio'] = (audio_df['valence'] > 0.6).mean()
        features['acoustic_preference'] = (audio_df['acousticness'] > 0.5).mean()
        features['mellow_preference'] = ((audio_df['valence'] > 0.4) & (audio_df['energy'] < 0.6)).mean()
        
        # Avoid aggressive music
        features['avoid_aggressive'] = 1 - ((audio_df['energy'] > 0.8) & (audio_df['loudness'] > -10)).mean()
        
        # === NEUROTICISM FEATURES ===
        features['emotional_variance'] = audio_df['valence'].std()
        features['energy_variance'] = audio_df['energy'].std()
        features['low_valence_ratio'] = (audio_df['valence'] < 0.4).mean()
        features['sad_music_preference'] = (audio_df['valence'] < 0.3).mean()
        
        # Extreme emotions (very high or very low valence)
        features['extreme_emotions'] = ((audio_df['valence'] < 0.2) | (audio_df['valence'] > 0.8)).mean()
        
        # === ADDITIONAL BEHAVIORAL FEATURES ===
        # Artist diversity
        all_artists = []
        for source in ['top_artists_short', 'top_artists_medium', 'top_artists_long']:
            if source in user_data:
                for artist in user_data[source]['items']:
                    all_artists.append(artist['name'])
        
        features['artist_diversity'] = len(set(all_artists)) / len(all_artists) if all_artists else 0
        
        # Time range consistency (do preferences change over time?)
        if all(['top_tracks_short', 'top_tracks_medium', 'top_tracks_long'] in user_data for _ in range(3)):
            short_valence = np.mean([t['valence'] for t in user_data.get('short_audio_features', []) if t])
            long_valence = np.mean([t['valence'] for t in user_data.get('long_audio_features', []) if t])
            features['preference_stability'] = 1 - abs(short_valence - long_valence) if short_valence and long_valence else 0.5
        else:
            features['preference_stability'] = 0.5
        
        # Handle any NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.5  # Neutral default
        
        return pd.Series(features)
    
    def predict_personality(self, behavioral_features):
        """Predict personality traits from behavioral features"""
        if self.models is None or behavioral_features is None:
            return None
        
        predictions = {}
        confidence_scores = {}
        
        # Ensure we have all required features
        model_features = self.models['performance_results'][list(self.models['models'].keys())[0]]['selected_features']
        
        # Create feature vector with defaults for missing features
        feature_vector = []
        for feature_name in model_features:
            if feature_name in behavioral_features:
                feature_vector.append(behavioral_features[feature_name])
            else:
                feature_vector.append(0.5)  # Neutral default
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Predict each trait
        for trait_name, model in self.models['models'].items():
            scaler = self.models['scalers'][trait_name]
            
            # Scale features
            scaled_features = scaler.transform(feature_array)
            
            # Predict
            prediction = model.predict(scaled_features)[0]
            
            # Clip to valid range (1-5 for Big Five)
            prediction = np.clip(prediction, 1.0, 5.0)
            
            predictions[trait_name] = round(prediction, 2)
            
            # Calculate confidence (inverse of model error)
            model_perf = self.models['performance_results'][trait_name]
            confidence = min(0.95, max(0.1, model_perf['test_r2']))
            confidence_scores[trait_name] = round(confidence, 2)
        
        return predictions, confidence_scores
    
    def generate_personality_insights(self, predictions, confidence_scores):
        """Generate human-readable personality insights"""
        insights = {}
        
        trait_descriptions = {
            'Openness': {
                'high': "You're curious and open to new experiences. You enjoy exploring different music genres and discovering new artists.",
                'medium': "You balance familiar favorites with occasional musical exploration.",
                'low': "You prefer familiar music and stick to genres you know and love."
            },
            'Conscientiousness': {
                'high': "Your music habits are consistent and organized. You likely have well-curated playlists and regular listening routines.",
                'medium': "You have some structure in your music preferences but also enjoy spontaneous discoveries.",
                'low': "Your musical tastes are more spontaneous and varied from day to day."
            },
            'Extraversion': {
                'high': "You gravitate toward energetic, social music. You probably enjoy music that gets you moving and would be great at parties.",
                'medium': "You enjoy a mix of high-energy and mellow music depending on your mood and situation.",
                'low': "You prefer quieter, more introspective music that's perfect for personal reflection."
            },
            'Agreeableness': {
                'high': "You prefer harmonious, positive music. You likely enjoy mainstream hits and music that brings people together.",
                'medium': "You appreciate both uplifting and more complex emotional music.",
                'low': "You're drawn to more intense or unconventional music that others might find challenging."
            },
            'Neuroticism': {
                'high': "Your music reflects emotional depth and complexity. You may use music to process feelings and find comfort.",
                'medium': "You enjoy both emotionally rich and stable, calming music.",
                'low': "You prefer stable, positive music that maintains good vibes."
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
                'description': trait_descriptions[trait][category],
                'confidence': confidence_scores.get(trait, 0.5)
            }
        
        return insights

def create_streamlit_app():
    """Create Streamlit web app interface"""
    st.set_page_config(
        page_title="Spotify Personality Predictor",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Spotify Personality Predictor")
    st.markdown("Discover your Big Five personality traits based on your Spotify listening habits!")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Input Spotify API credentials
    client_id = st.sidebar.text_input("Spotify Client ID", type="password")
    client_secret = st.sidebar.text_input("Spotify Client Secret", type="password")
    redirect_uri = st.sidebar.text_input("Redirect URI", value="http://localhost:8080/callback")
    
    if client_id and client_secret:
        try:
            # Initialize app
            app = SpotifyPersonalityApp(client_id, client_secret, redirect_uri)
            
            if st.sidebar.button("Analyze My Music"):
                with st.spinner("Fetching your Spotify data..."):
                    # Get user data
                    user_data = app.get_comprehensive_user_data()
                    
                    if user_data:
                        st.success("Data fetched successfully!")
                        
                        # Extract features
                        with st.spinner("Analyzing your music preferences..."):
                            features = app.extract_behavioral_features(user_data)
                            
                            if features is not None:
                                # Predict personality
                                predictions, confidence = app.predict_personality(features)
                                
                                if predictions:
                                    # Generate insights
                                    insights = app.generate_personality_insights(predictions, confidence)
                                    
                                    # Display results
                                    st.header("🎯 Your Personality Profile")
                                    
                                    # Create radar chart
                                    fig = go.Figure()
                                    
                                    traits = list(predictions.keys())
                                    scores = list(predictions.values())
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=scores,
                                        theta=traits,
                                        fill='toself',
                                        name='Your Personality',
                                        line_color='rgb(34, 139, 34)'
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[1, 5]
                                            )),
                                        showlegend=True,
                                        title="Your Big Five Personality Traits",
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display detailed insights
                                    st.header("📝 Detailed Insights")
                                    
                                    for trait, insight in insights.items():
                                        with st.expander(f"{trait}: {insight['score']}/5.0"):
                                            st.write(insight['description'])
                                            st.progress(insight['confidence'])
                                            st.caption(f"Confidence: {insight['confidence']:.1%}")
                                    
                                    # Show some music stats
                                    st.header("🎼 Your Music Stats")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Average Energy", f"{features['avg_energy']:.2f}")
                                        st.metric("Danceability", f"{features['avg_danceability']:.2f}")
                                    
                                    with col2:
                                        st.metric("Positivity", f"{features['avg_valence']:.2f}")
                                        st.metric("Acousticness", f"{features['avg_acousticness']:.2f}")
                                    
                                    with col3:
                                        st.metric("Musical Diversity", f"{features['key_diversity']:.2f}")
                                        st.metric("Artist Variety", f"{features['artist_diversity']:.2f}")
                                
                                else:
                                    st.error("Could not generate personality predictions. Please check the model.")
                            else:
                                st.error("Could not extract features from your music data.")
                    else:
                        st.error("Could not fetch your Spotify data. Please check your API credentials and permissions.")
        
        except Exception as e:
            st.error(f"Error initializing app: {e}")
    
    else:
        st.info("Please enter your Spotify API credentials in the sidebar to get started.")
        
        with st.expander("How to get Spotify API credentials"):
            st.markdown("""
            1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
            2. Log in with your Spotify account
            3. Click "Create App"
            4. Fill in app details (name, description)
            5. Set redirect URI to: `http://localhost:8080/callback`
            6. Copy your Client ID and Client Secret
            """)

if __name__ == "__main__":
    create_streamlit_app()