import os
import time
import hashlib
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Credentials
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(page_title="Auth Debug", page_icon="🔧", layout="wide")

def get_spotify_client():
    """Authenticate a user and return a Spotipy client and user profile.
       Uses a unique cache file per Streamlit session to avoid token leaks.
    """

    # Assign a unique session ID if this is the first run in this tab
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(int(time.time() * 1000))

    session_id = st.session_state.session_id
    cache_path = f".cache-{session_id}"

    # Check for auth code returned from Spotify
    params = st.query_params
    if "code" in params:
        code = params["code"]
        st.write(f"DEBUG: Got auth code: {code[:20]}...")

        auth_manager = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-top-read user-read-recently-played",
            cache_path=cache_path,       # unique cache per session
            show_dialog=True             # always ask user to approve
        )

        try:
            token_info = auth_manager.get_access_token(code, as_dict=True)
            if token_info and "access_token" in token_info:
                # Create client
                sp = spotipy.Spotify(auth=token_info["access_token"])
                user = sp.current_user()
                st.success(f"Token belongs to: {user.get('display_name', 'Unknown')} ({user.get('id', 'Unknown')})")

                # Clear query parameters to prevent re-use
                st.query_params.clear()

                # Save token info to session state for potential reuse
                st.session_state.token_info = token_info

                return sp, user
            else:
                st.error("No access token in response")
                return None, None
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
            return None, None

    # No code yet – generate a login URL
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-top-read user-read-recently-played",
        cache_path=cache_path,
        show_dialog=True,
        state=f"debug_{session_id}"     # unique state per session
    )
    login_url = auth_manager.get_authorize_url()
    st.markdown(f"[Login with Spotify]({login_url})")

    return None, None

def test_data_collection(sp, user):
    """Fetch recent and top tracks for the authenticated user and record a fingerprint."""
    st.write("Testing data collection...")

    try:
        # Recent tracks
        recent = sp.current_user_recently_played(limit=5)
        recent_tracks = [item["track"]["name"] for item in recent["items"]]

        # Top tracks
        top = sp.current_user_top_tracks(limit=5, time_range="short_term")
        top_tracks = [track["name"] for track in top["items"]]

        # Create a simple fingerprint
        all_tracks = recent_tracks + top_tracks
        fingerprint_data = f"{user['id']}_{len(all_tracks)}_{'_'.join(all_tracks)}"
        fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:8]

        # Display results
        st.subheader(f"Data for: {user['display_name']} ({user['id']})")
        st.write(f"**Data fingerprint:** {fingerprint}")

        st.write("**Recent tracks:**")
        for track in recent_tracks:
            st.write(f"- {track}")

        st.write("**Top tracks:**")
        for track in top_tracks:
            st.write(f"- {track}")

        # Store in session state for comparison
        if "test_results" not in st.session_state:
            st.session_state.test_results = []

        result = {
            "user_id": user["id"],
            "display_name": user["display_name"],
            "fingerprint": fingerprint,
            "recent_tracks": recent_tracks,
            "top_tracks": top_tracks,
        }
        st.session_state.test_results.append(result)

        return result

    except Exception as e:
        st.error(f"Data collection failed: {e}")
        return None

def show_comparison():
    """Compare fingerprints between different users."""
    if "test_results" not in st.session_state or len(st.session_state.test_results) < 2:
        st.info("Need at least 2 users to compare results")
        return

    st.header("Comparison Results")
    results = st.session_state.test_results

    for i, result in enumerate(results):
        st.write(f"**User {i + 1}:** {result['display_name']} ({result['user_id']}) - Fingerprint: {result['fingerprint']}")

    fingerprints = [r["fingerprint"] for r in results]
    user_ids = [r["user_id"] for r in results]

    if len(set(fingerprints)) == len(fingerprints):
        st.success("SUCCESS: All users have unique fingerprints")
    else:
        st.error("FAILURE: Duplicate fingerprints detected")

    if len(set(user_ids)) == len(user_ids):
        st.success("SUCCESS: All different user IDs")
    else:
        st.error("FAILURE: Duplicate user IDs detected")

    # Detailed view
    st.subheader("Detailed Data")
    for i, result in enumerate(results):
        with st.expander(f"User {i + 1}: {result['display_name']}"):
            st.write("Recent tracks:", result["recent_tracks"])
            st.write("Top tracks:", result["top_tracks"])

def main():
    st.title("Minimal Authentication Debug")
    st.write("Testing if different users get different Spotify data")

    if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
        st.error("Missing Spotify credentials")
        st.stop()

    # Buttons to clear session state
    if st.button("Reset (Keep Results)"):
        keys_to_keep = ["test_results", "session_id"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()

    if st.button("Clear All Results"):
        st.session_state.clear()
        st.rerun()

    # Authenticate user
    sp, user = get_spotify_client()

    # If authenticated, allow data collection
    if sp and user:
        if st.button("Test Data Collection"):
            result = test_data_collection(sp, user)
            if result:
                st.success("Data collection complete")

    # Show comparison across users
    show_comparison()

    # Instructions
    st.markdown("---")
    st.subheader("Test Instructions")
    st.write("1. You: Click login, authenticate, test data collection")
    st.write("2. Friend: Open NEW incognito window, repeat same steps")
    st.write("3. Compare fingerprints – should be different")

    if "test_results" in st.session_state:
        st.write(f"Current results: {len(st.session_state.test_results)} users tested")

if __name__ == "__main__":
    main()
