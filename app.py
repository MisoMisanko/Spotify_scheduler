# app.py - USER-ISOLATED DEBUG VERSION
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
    page_title="Spotify Debug - User Isolated",
    page_icon="🔐",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

# CRITICAL: Generate unique session ID to prevent cross-user contamination
if 'session_uuid' not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())
    st.session_state.session_start = datetime.now().isoformat()

def clear_all_caches():
    """Force clear all Streamlit caches to prevent data leakage"""
    st.cache_data.clear()
    st.cache_resource.clear()

def get_auth_manager():
    """Create a fresh auth manager with unique state parameter"""
    # Use session UUID to ensure unique auth state
    unique_state = f"spotify_auth_{st.session_state.session_uuid}"
    
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        open_browser=False,
        cache_path=None,  # NEVER cache tokens
        state=unique_state,  # Unique state per session
        show_dialog=True  # Force login dialog every time
    )

def ensure_spotify_client():
    """Ensure fresh Spotify client with NO session state reuse"""
    
    # Display session info for debugging
    st.sidebar.write(f"🔐 Session ID: {st.session_state.session_uuid[:8]}...")
    st.sidebar.write(f"🔐 Session Start: {st.session_state.session_start}")
    
    auth_manager = get_auth_manager()
    
    # Check for authorization code in URL (fresh login)
    params = st.query_params
    if "code" in params:
        code = params["code"]
        state = params.get("state", "")
        
        # Verify state matches our session
        expected_state = f"spotify_auth_{st.session_state.session_uuid}"
        if state != expected_state:
            st.error("🔐 DEBUG: State mismatch - potential session contamination!")
            st.write(f"Expected: {expected_state}")
            st.write(f"Received: {state}")
            st.stop()
        
        try:
            # Get fresh token
            token_info = auth_manager.get_access_token(code, as_dict=True)
            
            # Store token with session-specific key
            token_key = f"token_{st.session_state.session_uuid}"
            st.session_state[token_key] = token_info
            
            st.query_params.clear()
            st.success("🔐 Fresh authentication successful!")
            st.rerun()
            
        except Exception as e:
            st.error(f"🔐 Authentication error: {e}")
            st.stop()
    
    # Check for existing valid token for THIS session only
    token_key = f"token_{st.session_state.session_uuid}"
    token_info = st.session_state.get(token_key)
    
    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])
    
    # Need fresh login
    login_url = auth_manager.get_authorize_url()
    
    st.warning("🔐 Fresh authentication required")
    st.info("Each user must log in separately to prevent data mixing")
    st.markdown(f"[🔐 Login with YOUR Spotify Account]({login_url})")
    
    # Show clear separation
    st.markdown("---")
    st.error("⚠️ DO NOT proceed until YOU have logged in with YOUR account")
    st.stop()

def get_user_profile_debug(sp):
    """Get user profile to verify identity"""
    try:
        user_profile = sp.current_user()
        user_id = user_profile.get('id', 'unknown')
        display_name = user_profile.get('display_name', 'Unknown User')
        followers = user_profile.get('followers', {}).get('total', 0)
        
        return {
            'user_id': user_id,
            'display_name': display_name,
            'followers': followers,
            'profile_url': user_profile.get('external_urls', {}).get('spotify', ''),
            'country': user_profile.get('country', 'Unknown')
        }
    except Exception as e:
        st.error(f"🔐 Could not get user profile: {e}")
        return None

def get_user_music_data_isolated(sp, limit=20):
    """Get user's music data with complete isolation"""
    
    # First, verify user identity
    user_profile = get_user_profile_debug(sp)
    if not user_profile:
        return None
    
    st.success(f"🔐 Authenticated as: {user_profile['display_name']} ({user_profile['user_id']})")
    
    music_data = {
        'session_id': st.session_state.session_uuid,
        'user_profile': user_profile,
        'collection_time': datetime.now().isoformat(),
        'tracks': [],
        'artists': [],
        'genres': [],
        'raw_data': {}
    }
    
    try:
        st.write(f"🔐 Collecting data for {user_profile['display_name']}...")
        
        all_tracks = []
        
        # Get top tracks with detailed logging
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                st.write(f"  📊 Getting {time_range} tracks...")
                tracks_response = sp.current_user_top_tracks(limit=limit, time_range=time_range)
                tracks = tracks_response['items']
                
                if tracks:
                    all_tracks.extend(tracks)
                    # Show proof of different data
                    sample_tracks = [f"{t['name']} - {t['artists'][0]['name']}" for t in tracks[:3]]
                    st.write(f"    Sample tracks: {sample_tracks}")
                else:
                    st.write(f"    No {time_range} tracks found")
                    
            except Exception as e:
                st.warning(f"  ❌ Error getting {time_range} tracks: {e}")
        
        # Get recent tracks
        try:
            st.write("  📊 Getting recently played...")
            recent_response = sp.current_user_recently_played(limit=limit)
            recent_tracks = [item['track'] for item in recent_response['items']]
            
            if recent_tracks:
                all_tracks.extend(recent_tracks)
                sample_recent = [f"{t['name']} - {t['artists'][0]['name']}" for t in recent_tracks[:3]]
                st.write(f"    Recent tracks: {sample_recent}")
            else:
                st.write("    No recent tracks found")
                
        except Exception as e:
            st.warning(f"  ❌ Error getting recent tracks: {e}")
        
        if not all_tracks:
            st.error("🔐 No tracks found for this user!")
            return None
        
        # Remove duplicates
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track['id'] and track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        music_data['tracks'] = unique_tracks
        music_data['raw_data']['track_count'] = len(unique_tracks)
        
        st.success(f"🔐 Found {len(unique_tracks)} unique tracks for {user_profile['display_name']}")
        
        # Get artist info
        artist_ids = list(set([track['artists'][0]['id'] for track in unique_tracks]))
        st.write(f"  📊 Getting artist data for {len(artist_ids)} artists...")
        
        all_artists = []
        all_genres = []
        
        for i in range(0, len(artist_ids), 20):
            batch_ids = artist_ids[i:i+20]
            try:
                artists_response = sp.artists(batch_ids)
                batch_artists = artists_response['artists']
                all_artists.extend(batch_artists)
                
                for artist in batch_artists:
                    artist_genres = artist.get('genres', [])
                    all_genres.extend(artist_genres)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                st.warning(f"  ❌ Error getting artist batch: {e}")
        
        music_data['artists'] = all_artists
        music_data['genres'] = all_genres
        music_data['raw_data']['genre_count'] = len(all_genres)
        music_data['raw_data']['unique_genres'] = len(set(all_genres))
        
        # Show genre sample to prove different data
        if all_genres:
            unique_genres = list(set(all_genres))[:10]
            st.write(f"  📊 Sample genres: {unique_genres}")
        
        # Create a hash of the data to verify uniqueness
        data_string = f"{user_profile['user_id']}_{len(unique_tracks)}_{len(all_genres)}_{'_'.join(sorted(set(all_genres)))}"
        data_hash = hashlib.md5(data_string.encode()).hexdigest()[:8]
        music_data['data_fingerprint'] = data_hash
        
        st.success(f"🔐 Data fingerprint: {data_hash}")
        
        return music_data
        
    except Exception as e:
        st.error(f"🔐 Error collecting music data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def analyze_user_data_isolated(music_data):
    """Analyze user data with complete transparency"""
    
    if not music_data or not music_data['tracks']:
        st.error("🔐 No data to analyze!")
        return None
    
    user_profile = music_data['user_profile']
    tracks = music_data['tracks']
    genres = music_data['genres']
    
    st.header(f"🔐 Analysis for {user_profile['display_name']}")
    st.write(f"**User ID:** {user_profile['user_id']}")
    st.write(f"**Data Fingerprint:** {music_data['data_fingerprint']}")
    st.write(f"**Collection Time:** {music_data['collection_time']}")
    
    # Basic stats
    track_names = [track['name'] for track in tracks]
    artist_names = [track['artists'][0]['name'] for track in tracks]
    popularities = [track.get('popularity', 50) for track in tracks]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tracks Analyzed", len(tracks))
        st.metric("Unique Artists", len(set(artist_names)))
        
    with col2:
        st.metric("Total Genres", len(genres))
        st.metric("Unique Genres", len(set(genres)))
        
    with col3:
        avg_popularity = np.mean(popularities)
        st.metric("Avg Popularity", f"{avg_popularity:.1f}")
        st.metric("Mainstream %", f"{(avg_popularity > 70) * 100:.0f}%")
    
    # Show actual track data to prove uniqueness
    st.subheader("📊 Your Top Tracks")
    track_display = pd.DataFrame({
        'Track': track_names[:10],
        'Artist': artist_names[:10],
        'Popularity': popularities[:10]
    })
    st.dataframe(track_display)
    
    # Show genre distribution
    if genres:
        st.subheader("📊 Your Genres")
        genre_counts = pd.Series(genres).value_counts().head(10)
        st.bar_chart(genre_counts)
    
    # Calculate some basic features for demonstration
    features = {
        'avg_popularity': avg_popularity / 100,
        'mainstream_preference': sum(1 for p in popularities if p > 70) / len(popularities),
        'underground_preference': sum(1 for p in popularities if p < 30) / len(popularities),
        'genre_diversity': len(set(genres)) / max(len(genres), 1) if genres else 0,
        'unique_artist_ratio': len(set(artist_names)) / len(tracks)
    }
    
    st.subheader("📊 Your Music Features")
    features_df = pd.DataFrame([features]).T
    features_df.columns = ['Value']
    st.dataframe(features_df)
    
    # Store results with session isolation
    results_key = f"results_{st.session_state.session_uuid}"
    st.session_state[results_key] = {
        'user_id': user_profile['user_id'],
        'display_name': user_profile['display_name'],
        'features': features,
        'data_fingerprint': music_data['data_fingerprint'],
        'analysis_time': datetime.now().isoformat()
    }
    
    return features

def compare_sessions():
    """Show comparison between different sessions to verify isolation"""
    
    st.header("🔐 Session Isolation Verification")
    
    # Find all result sessions
    result_sessions = {}
    for key in st.session_state.keys():
        if key.startswith('results_'):
            session_id = key.replace('results_', '')
            result_sessions[session_id] = st.session_state[key]
    
    if len(result_sessions) > 1:
        st.success(f"✅ Found {len(result_sessions)} different user sessions")
        
        comparison_data = []
        for session_id, results in result_sessions.items():
            comparison_data.append({
                'Session': session_id[:8],
                'User': results['display_name'],
                'User ID': results['user_id'],
                'Data Fingerprint': results['data_fingerprint'],
                'Mainstream Pref': f"{results['features']['mainstream_preference']:.2f}",
                'Genre Diversity': f"{results['features']['genre_diversity']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Check for identical results (this should NOT happen)
        fingerprints = [r['data_fingerprint'] for r in result_sessions.values()]
        if len(set(fingerprints)) == 1:
            st.error("🚨 CRITICAL: All users have identical data fingerprints!")
        else:
            st.success("✅ Users have different data fingerprints - isolation working!")
            
    else:
        st.info("Only one user session found so far")

def main():
    st.title("🔐 Spotify Debug - User Data Isolation Test")
    st.markdown("### Ensuring each user sees only their own data")
    
    # Force clear caches on app start
    if st.button("🔄 Clear All Caches & Reset"):
        clear_all_caches()
        # Clear all session state except UUID
        keys_to_keep = ['session_uuid', 'session_start']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()
    
    # Show session isolation info
    with st.sidebar:
        st.header("🔐 Session Info")
        st.write(f"Session: {st.session_state.session_uuid[:8]}...")
        st.write(f"Started: {st.session_state.session_start}")
        
        # Show how many sessions we've seen
        session_count = len([k for k in st.session_state.keys() if k.startswith('results_')])
        st.write(f"Sessions analyzed: {session_count}")
    
    # Authentication with forced isolation
    sp = ensure_spotify_client()
    
    if st.button("🔐 Analyze MY Music Data", type="primary"):
        
        with st.spinner("🔐 Collecting YOUR music data (isolated)..."):
            music_data = get_user_music_data_isolated(sp)
        
        if music_data:
            with st.spinner("🔐 Analyzing YOUR data..."):
                features = analyze_user_data_isolated(music_data)
            
            if features:
                st.success("🔐 Analysis complete!")
                
                # Show session comparison if multiple users
                compare_sessions()
        else:
            st.error("🔐 Failed to collect music data")
    
    # Show instructions
    st.markdown("---")
    st.subheader("🔐 Testing Instructions")
    st.markdown("""
    1. **First User**: Click "Analyze MY Music Data" and login
    2. **Second User**: In a NEW browser tab/window, visit this same URL
    3. **Second User**: Click "Analyze MY Music Data" and login with DIFFERENT account
    4. **Compare**: Both users should see different data and fingerprints
    
    **Expected Result**: Each user sees only their own Spotify data
    **Bug Indicator**: If data fingerprints are identical, there's still cross-contamination
    """)

if __name__ == "__main__":
    main()