# app.py - USER ISOLATION VERIFICATION TEST
import os
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib
import json
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
    page_title="Spotify Isolation Test",
    page_icon="✅",
    layout="wide"
)

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        open_browser=False,
        cache_path=None,
        show_dialog=True
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
        st.query_params.clear()
        st.rerun()

    login_url = auth_manager.get_authorize_url()
    st.info("Please log in with Spotify")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

def get_comprehensive_user_data(sp):
    """Get comprehensive user data to verify uniqueness"""
    
    try:
        # Get user profile first
        user_profile = sp.current_user()
        user_id = user_profile.get('id', 'unknown')
        display_name = user_profile.get('display_name', 'Unknown')
        
        st.success(f"✅ Authenticated as: {display_name} ({user_id})")
        
        # Initialize data structure
        user_data = {
            'user_profile': {
                'id': user_id,
                'display_name': display_name,
                'followers': user_profile.get('followers', {}).get('total', 0),
                'country': user_profile.get('country', 'Unknown')
            },
            'top_tracks': {'short': [], 'medium': [], 'long': []},
            'top_artists': {'short': [], 'medium': [], 'long': []},
            'recent_tracks': [],
            'saved_tracks': [],
            'playlists': [],
            'following': [],
            'collection_timestamp': datetime.now().isoformat()
        }
        
        st.write("📊 Collecting comprehensive data...")
        
        # Get top tracks across all time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                tracks = sp.current_user_top_tracks(limit=20, time_range=time_range)
                user_data['top_tracks'][time_range.split('_')[0]] = tracks['items']
                st.write(f"  ✅ {time_range} tracks: {len(tracks['items'])}")
            except Exception as e:
                st.write(f"  ❌ {time_range} tracks error: {e}")
        
        # Get top artists across all time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            try:
                artists = sp.current_user_top_artists(limit=20, time_range=time_range)
                user_data['top_artists'][time_range.split('_')[0]] = artists['items']
                st.write(f"  ✅ {time_range} artists: {len(artists['items'])}")
            except Exception as e:
                st.write(f"  ❌ {time_range} artists error: {e}")
        
        # Get recently played
        try:
            recent = sp.current_user_recently_played(limit=50)
            user_data['recent_tracks'] = [item['track'] for item in recent['items']]
            st.write(f"  ✅ Recent tracks: {len(user_data['recent_tracks'])}")
        except Exception as e:
            st.write(f"  ❌ Recent tracks error: {e}")
        
        # Get saved tracks (liked songs)
        try:
            saved = sp.current_user_saved_tracks(limit=50)
            user_data['saved_tracks'] = [item['track'] for item in saved['items']]
            st.write(f"  ✅ Saved tracks: {len(user_data['saved_tracks'])}")
        except Exception as e:
            st.write(f"  ❌ Saved tracks error: {e}")
        
        # Get playlists
        try:
            playlists = sp.current_user_playlists(limit=20)
            user_data['playlists'] = playlists['items']
            st.write(f"  ✅ Playlists: {len(user_data['playlists'])}")
        except Exception as e:
            st.write(f"  ❌ Playlists error: {e}")
        
        # Get following
        try:
            following = sp.current_user_followed_artists(limit=20)
            user_data['following'] = following['artists']['items']
            st.write(f"  ✅ Following: {len(user_data['following'])}")
        except Exception as e:
            st.write(f"  ❌ Following error: {e}")
        
        return user_data
        
    except Exception as e:
        st.error(f"Error collecting user data: {e}")
        return None

def create_user_signature(user_data):
    """Create a unique signature from user's actual Spotify data"""
    
    if not user_data:
        return None
    
    # Extract key identifying information
    profile = user_data['user_profile']
    
    # Get track names from all sources
    all_track_names = []
    
    # From top tracks
    for period in user_data['top_tracks'].values():
        all_track_names.extend([track['name'] for track in period])
    
    # From recent and saved
    all_track_names.extend([track['name'] for track in user_data['recent_tracks']])
    all_track_names.extend([track['name'] for track in user_data['saved_tracks']])
    
    # Get artist names from all sources
    all_artist_names = []
    
    # From top artists
    for period in user_data['top_artists'].values():
        all_artist_names.extend([artist['name'] for artist in period])
    
    # From tracks
    for period in user_data['top_tracks'].values():
        all_artist_names.extend([track['artists'][0]['name'] for track in period])
    
    for track in user_data['recent_tracks'] + user_data['saved_tracks']:
        if track['artists']:
            all_artist_names.append(track['artists'][0]['name'])
    
    # Get playlist names
    playlist_names = [playlist['name'] for playlist in user_data['playlists']]
    
    # Get following names
    following_names = [artist['name'] for artist in user_data['following']]
    
    # Create comprehensive signature
    signature_data = {
        'user_id': profile['id'],
        'display_name': profile['display_name'],
        'followers': profile['followers'],
        'country': profile['country'],
        'unique_tracks': len(set(all_track_names)),
        'unique_artists': len(set(all_artist_names)),
        'total_tracks': len(all_track_names),
        'total_artists': len(all_artist_names),
        'playlists_count': len(playlist_names),
        'following_count': len(following_names),
        'top_tracks_sample': sorted(list(set(all_track_names)))[:10],
        'top_artists_sample': sorted(list(set(all_artist_names)))[:10],
        'playlist_sample': playlist_names[:5],
        'following_sample': following_names[:5]
    }
    
    # Create hash of the signature
    signature_string = json.dumps(signature_data, sort_keys=True)
    signature_hash = hashlib.sha256(signature_string.encode()).hexdigest()[:16]
    
    return {
        'signature_hash': signature_hash,
        'signature_data': signature_data,
        'raw_data_sample': {
            'tracks': all_track_names[:5],
            'artists': all_artist_names[:5],
            'playlists': playlist_names[:3]
        }
    }

def display_user_analysis(user_data, signature):
    """Display comprehensive user analysis"""
    
    st.header("🔍 User Data Analysis")
    
    profile = user_data['user_profile']
    sig_data = signature['signature_data']
    
    # User identity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 User Identity")
        st.write(f"**Display Name:** {profile['display_name']}")
        st.write(f"**User ID:** {profile['id']}")
        st.write(f"**Followers:** {profile['followers']:,}")
        st.write(f"**Country:** {profile['country']}")
        st.write(f"**Data Signature:** `{signature['signature_hash']}`")
    
    with col2:
        st.subheader("📊 Music Library Stats")
        st.metric("Unique Tracks Found", sig_data['unique_tracks'])
        st.metric("Unique Artists Found", sig_data['unique_artists'])
        st.metric("Total Playlists", sig_data['playlists_count'])
        st.metric("Artists Following", sig_data['following_count'])
    
    # Show actual data samples
    st.subheader("🎵 Your Actual Music Data (Proof of Uniqueness)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Top Tracks", "Top Artists", "Playlists", "Following"])
    
    with tab1:
        if sig_data['top_tracks_sample']:
            st.write("Sample of your top tracks:")
            for i, track in enumerate(sig_data['top_tracks_sample'], 1):
                st.write(f"{i}. {track}")
        else:
            st.write("No top tracks found")
    
    with tab2:
        if sig_data['top_artists_sample']:
            st.write("Sample of your top artists:")
            for i, artist in enumerate(sig_data['top_artists_sample'], 1):
                st.write(f"{i}. {artist}")
        else:
            st.write("No top artists found")
    
    with tab3:
        if sig_data['playlist_sample']:
            st.write("Sample of your playlists:")
            for i, playlist in enumerate(sig_data['playlist_sample'], 1):
                st.write(f"{i}. {playlist}")
        else:
            st.write("No playlists found")
    
    with tab4:
        if sig_data['following_sample']:
            st.write("Sample of artists you follow:")
            for i, artist in enumerate(sig_data['following_sample'], 1):
                st.write(f"{i}. {artist}")
        else:
            st.write("No followed artists found")
    
    return signature

def compare_user_signatures():
    """Compare signatures from different users to verify isolation"""
    
    if 'user_signatures' not in st.session_state:
        st.session_state.user_signatures = []
    
    signatures = st.session_state.user_signatures
    
    if len(signatures) > 1:
        st.header("🔍 Multi-User Comparison")
        
        # Create comparison table
        comparison_data = []
        for sig in signatures:
            data = sig['signature_data']
            comparison_data.append({
                'User ID': data['user_id'],
                'Display Name': data['display_name'],
                'Signature Hash': sig['signature_hash'],
                'Unique Tracks': data['unique_tracks'],
                'Unique Artists': data['unique_artists'],
                'Playlists': data['playlists_count'],
                'Following': data['following_count'],
                'Collection Time': sig.get('timestamp', 'Unknown')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Check for isolation
        hashes = [sig['signature_hash'] for sig in signatures]
        user_ids = [sig['signature_data']['user_id'] for sig in signatures]
        
        st.subheader("🔍 Isolation Verification")
        
        if len(set(hashes)) == len(hashes):
            st.success("✅ SUCCESS: All users have unique data signatures!")
            st.success("✅ User isolation is working correctly!")
        else:
            st.error("❌ FAILURE: Duplicate signatures detected!")
            st.error("❌ Users are seeing identical data - isolation failed!")
        
        if len(set(user_ids)) == len(user_ids):
            st.success("✅ All different user IDs confirmed")
        else:
            st.error("❌ Duplicate user IDs detected!")
        
        # Show detailed differences
        st.subheader("🔍 Detailed Differences")
        
        for i, sig in enumerate(signatures):
            with st.expander(f"User {i+1}: {sig['signature_data']['display_name']}"):
                st.json(sig['raw_data_sample'])
    
    else:
        st.info("Need at least 2 users to compare. Have your friend run this test too!")

def main():
    st.title("✅ User Isolation Verification Test")
    st.markdown("### Comprehensive test to verify each user sees only their own Spotify data")
    
    # Instructions
    with st.expander("📋 Test Instructions"):
        st.markdown("""
        **Step 1:** You run this test first
        **Step 2:** Your friend (added to your Spotify app) runs this test in a separate browser
        **Step 3:** Compare the signature hashes - they should be completely different
        
        **What this test checks:**
        - User profile information
        - Top tracks/artists across all time periods
        - Recently played tracks
        - Saved/liked tracks
        - User playlists
        - Followed artists
        
        **Expected result:** Different signature hashes = successful isolation
        """)
    
    sp = ensure_spotify_client()
    
    if st.button("🔍 Run Comprehensive Data Test", type="primary"):
        
        with st.spinner("Collecting comprehensive user data..."):
            user_data = get_comprehensive_user_data(sp)
        
        if user_data:
            with st.spinner("Creating user signature..."):
                signature = create_user_signature(user_data)
            
            if signature:
                # Add timestamp
                signature['timestamp'] = datetime.now().isoformat()
                
                # Display analysis
                display_user_analysis(user_data, signature)
                
                # Store signature
                if 'user_signatures' not in st.session_state:
                    st.session_state.user_signatures = []
                
                st.session_state.user_signatures.append(signature)
                
                # Compare with other users
                compare_user_signatures()
                
                st.success("✅ Test complete! Check the comparison section above.")
        
        else:
            st.error("❌ Failed to collect user data")
    
    # Show current signatures
    if 'user_signatures' in st.session_state and st.session_state.user_signatures:
        st.subheader("📝 Test Results Summary")
        signatures = st.session_state.user_signatures
        st.write(f"**Tests completed:** {len(signatures)}")
        
        for i, sig in enumerate(signatures):
            st.write(f"**User {i+1}:** {sig['signature_data']['display_name']} - Signature: `{sig['signature_hash']}`")

if __name__ == "__main__":
    main()