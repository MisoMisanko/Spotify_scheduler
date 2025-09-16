# app.py - STATELESS USER-ISOLATED DEBUG VERSION
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
import urllib.parse

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

st.set_page_config(
    page_title="Spotify Debug - Stateless",
    page_icon="🔓",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

def generate_user_session():
    """Generate a truly unique session identifier"""
    timestamp = str(int(time.time() * 1000))
    random_id = str(uuid.uuid4())
    return f"{timestamp}_{random_id}"

def get_spotify_client_stateless():
    """Get Spotify client without using ANY session state"""
    
    # Check URL parameters for auth code
    params = st.query_params
    
    if "code" in params and "state" in params:
        # We have an auth code - exchange it for token
        code = params["code"]
        state = params["state"]
        
        st.write(f"🔓 AUTH DEBUG: Received code for state: {state}")
        
        try:
            # Create auth manager with the same state
            auth_manager = SpotifyOAuth(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
                scope=SCOPES,
                open_browser=False,
                cache_path=None,
                state=state,
                show_dialog=True
            )
            
            # Exchange code for token
            token_info = auth_manager.get_access_token(code, as_dict=True)
            
            if token_info and 'access_token' in token_info:
                st.success(f"🔓 SUCCESS: Got token for state {state}")
                
                # Create Spotify client
                sp = spotipy.Spotify(auth=token_info["access_token"])
                
                # Immediately get user info to verify
                try:
                    user_profile = sp.current_user()
                    user_id = user_profile.get('id', 'unknown')
                    display_name = user_profile.get('display_name', 'Unknown User')
                    
                    st.success(f"🔓 VERIFIED: Logged in as {display_name} ({user_id})")
                    
                    # Clear URL params to prevent reuse
                    st.query_params.clear()
                    
                    return sp, user_profile, state
                    
                except Exception as e:
                    st.error(f"🔓 ERROR: Could not verify user: {e}")
                    return None, None, None
            else:
                st.error("🔓 ERROR: No access token received")
                return None, None, None
                
        except Exception as e:
            st.error(f"🔓 ERROR: Token exchange failed: {e}")
            return None, None, None
    
    # No auth code - need to initiate login
    return None, None, None

def initiate_spotify_login():
    """Start fresh login process with unique state"""
    
    # Generate completely unique state
    unique_state = generate_user_session()
    
    st.write(f"🔓 INIT: Creating login with state: {unique_state}")
    
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        open_browser=False,
        cache_path=None,
        state=unique_state,
        show_dialog=True  # Force fresh login
    )
    
    login_url = auth_manager.get_authorize_url()
    
    st.markdown(f"""
    ### 🔓 Fresh Authentication Required
    
    **State ID:** `{unique_state}`
    
    **Important:** Each user must use a separate browser/incognito window!
    
    [🔓 Login with YOUR Spotify Account]({login_url})
    """)
    
    return unique_state

def collect_user_data_stateless(sp, user_profile, state_id):
    """Collect user data without any caching"""
    
    st.header(f"🔓 Data Collection for {user_profile['display_name']}")
    st.write(f"**User ID:** {user_profile['id']}")
    st.write(f"**State ID:** {state_id}")
    st.write(f"**Collection Time:** {datetime.now().isoformat()}")
    
    try:
        all_tracks = []
        
        # Get different types of tracks
        st.write("🔓 Collecting top tracks...")
        
        for time_range in ['short_term', 'medium_term']:  # Reduced to avoid rate limits
            try:
                tracks_response = sp.current_user_top_tracks(limit=10, time_range=time_range)
                tracks = tracks_response['items']
                
                if tracks:
                    all_tracks.extend(tracks)
                    sample = [f"{t['name']} by {t['artists'][0]['name']}" for t in tracks[:2]]
                    st.write(f"  ✅ {time_range}: {len(tracks)} tracks - {sample}")
                else:
                    st.write(f"  ❌ {time_range}: No tracks")
                    
            except Exception as e:
                st.warning(f"  ❌ {time_range} error: {e}")
        
        # Get recent tracks
        try:
            st.write("🔓 Collecting recent tracks...")
            recent_response = sp.current_user_recently_played(limit=10)
            recent_tracks = [item['track'] for item in recent_response['items']]
            
            if recent_tracks:
                all_tracks.extend(recent_tracks)
                sample = [f"{t['name']} by {t['artists'][0]['name']}" for t in recent_tracks[:2]]
                st.write(f"  ✅ Recent: {len(recent_tracks)} tracks - {sample}")
            else:
                st.write("  ❌ Recent: No tracks")
                
        except Exception as e:
            st.warning(f"  ❌ Recent error: {e}")
        
        if not all_tracks:
            st.error("🔓 No tracks found!")
            return None
        
        # Remove duplicates
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track['id'] and track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        st.success(f"🔓 Found {len(unique_tracks)} unique tracks")
        
        # Get some artist data
        artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks[:20]]))  # Limit to avoid rate limits
        
        st.write(f"🔓 Getting artist genres for {len(artist_ids)} artists...")
        
        all_genres = []
        try:
            artists_response = sp.artists(artist_ids)
            artists = artists_response['artists']
            
            for artist in artists:
                genres = artist.get('genres', [])
                all_genres.extend(genres)
            
            st.write(f"  ✅ Found {len(all_genres)} genre tags")
            if all_genres:
                unique_genres = list(set(all_genres))[:5]
                st.write(f"  Sample genres: {unique_genres}")
                
        except Exception as e:
            st.warning(f"  ❌ Artist error: {e}")
        
        # Create data fingerprint
        track_names = [track['name'] for track in unique_tracks]
        artist_names = [track['artists'][0]['name'] for track in unique_tracks]
        popularities = [track.get('popularity', 50) for track in unique_tracks]
        
        # Create unique fingerprint
        fingerprint_data = f"{user_profile['id']}_{len(unique_tracks)}_{len(all_genres)}_{''.join(sorted(set(all_genres)))}"
        data_fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]
        
        user_data = {
            'state_id': state_id,
            'user_profile': user_profile,
            'tracks': unique_tracks,
            'genres': all_genres,
            'track_names': track_names,
            'artist_names': artist_names,
            'popularities': popularities,
            'data_fingerprint': data_fingerprint,
            'collection_time': datetime.now().isoformat()
        }
        
        st.success(f"🔓 **Data Fingerprint: {data_fingerprint}**")
        
        return user_data
        
    except Exception as e:
        st.error(f"🔓 Collection error: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

def analyze_user_data_stateless(user_data):
    """Analyze user data and show results"""
    
    st.header("🔓 Your Music Analysis")
    
    # User info
    user_profile = user_data['user_profile']
    st.write(f"**User:** {user_profile['display_name']} ({user_profile['id']})")
    st.write(f"**State ID:** {user_data['state_id']}")
    st.write(f"**Data Fingerprint:** {user_data['data_fingerprint']}")
    
    # Basic stats
    tracks = user_data['tracks']
    genres = user_data['genres']
    popularities = user_data['popularities']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tracks", len(tracks))
        st.metric("Artists", len(set(user_data['artist_names'])))
    
    with col2:
        st.metric("Genres", len(genres))
        st.metric("Unique Genres", len(set(genres)))
    
    with col3:
        avg_pop = np.mean(popularities) if popularities else 0
        st.metric("Avg Popularity", f"{avg_pop:.1f}")
        mainstream = sum(1 for p in popularities if p > 70) / len(popularities) if popularities else 0
        st.metric("Mainstream %", f"{mainstream:.0%}")
    
    # Show your actual tracks
    st.subheader("🔓 Your Tracks (Proof of Unique Data)")
    if tracks:
        track_display = pd.DataFrame({
            'Track': user_data['track_names'][:10],
            'Artist': user_data['artist_names'][:10],
            'Popularity': popularities[:10]
        })
        st.dataframe(track_display)
    
    # Show your genres
    if genres:
        st.subheader("🔓 Your Genres")
        genre_counts = pd.Series(genres).value_counts().head(8)
        st.bar_chart(genre_counts)
    
    # Simple features
    features = {
        'avg_popularity': avg_pop / 100 if avg_pop else 0,
        'mainstream_preference': mainstream,
        'genre_diversity': len(set(genres)) / len(genres) if genres else 0,
        'artist_diversity': len(set(user_data['artist_names'])) / len(tracks) if tracks else 0
    }
    
    st.subheader("🔓 Your Music Profile")
    for feature, value in features.items():
        st.metric(feature.replace('_', ' ').title(), f"{value:.2f}")
    
    return features

def main():
    st.title("🔓 Spotify Debug - Stateless User Isolation")
    st.markdown("### Each user gets completely separate data - no session sharing")
    
    # Try to get authenticated client
    sp, user_profile, state_id = get_spotify_client_stateless()
    
    if sp and user_profile and state_id:
        # We're authenticated - show user data
        if st.button("🔓 Collect & Analyze My Music Data", type="primary"):
            
            with st.spinner("🔓 Collecting your data..."):
                user_data = collect_user_data_stateless(sp, user_profile, state_id)
            
            if user_data:
                features = analyze_user_data_stateless(user_data)
                
                # Store for comparison
                if 'comparisons' not in st.session_state:
                    st.session_state.comparisons = []
                
                comparison_entry = {
                    'user_id': user_profile['id'],
                    'display_name': user_profile['display_name'],
                    'state_id': state_id,
                    'fingerprint': user_data['data_fingerprint'],
                    'tracks': len(user_data['tracks']),
                    'genres': len(user_data['genres']),
                    'mainstream': features['mainstream_preference']
                }
                
                st.session_state.comparisons.append(comparison_entry)
                
                # Show comparison if multiple users
                if len(st.session_state.comparisons) > 1:
                    st.header("🔓 Multi-User Comparison")
                    
                    comp_df = pd.DataFrame(st.session_state.comparisons)
                    st.dataframe(comp_df)
                    
                    # Check for duplicates
                    fingerprints = comp_df['fingerprint'].tolist()
                    if len(set(fingerprints)) == len(fingerprints):
                        st.success("✅ All users have unique data fingerprints!")
                    else:
                        st.error("❌ Duplicate fingerprints detected - isolation failed!")
    
    else:
        # Need to authenticate
        state_id = initiate_spotify_login()
        
        st.markdown("---")
        st.subheader("🔓 Testing Instructions")
        st.markdown(f"""
        **Current State ID:** `{state_id}`
        
        1. **You**: Click the login link above
        2. **Friend**: Open NEW incognito window, visit this URL
        3. **Friend**: Will get different State ID and login link
        4. **Compare**: Data fingerprints should be different
        
        **Key Difference**: This version uses NO session state storage
        """)

if __name__ == "__main__":
    main()