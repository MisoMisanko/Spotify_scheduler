import os
import json
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

SCOPES = "user-top-read user-read-recently-played user-library-read"
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(page_title="Spotify Data Exporter", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Data Exporter")

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
    )

def ensure_spotify_client():
    auth_manager = get_auth_manager()
    token_info = st.session_state.get("token_info")

    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0] if isinstance(params["code"], list) else params["code"]
        token_info = auth_manager.get_access_token(code, as_dict=True)
        st.session_state["token_info"] = token_info
        st.experimental_set_query_params()
        st.rerun()

    login_url = auth_manager.get_authorize_url()
    st.info("Please log in with Spotify")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

def fetch_data(sp):
    data = {}
    # Profile
    data["user"] = sp.current_user()

    # Top tracks/artists
    for r in ["short_term", "medium_term", "long_term"]:
        data[f"top_tracks_{r}"] = sp.current_user_top_tracks(limit=50, time_range=r)["items"]
        data[f"top_artists_{r}"] = sp.current_user_top_artists(limit=50, time_range=r)["items"]

    # Recently played
    data["recent"] = sp.current_user_recently_played(limit=50)["items"]

    # Saved tracks (first 100 for speed)
    data["saved_tracks"] = sp.current_user_saved_tracks(limit=100)["items"]

    return data

def main():
    sp = ensure_spotify_client()
    st.success("Spotify authenticated ✅")

    if st.button("📦 Export Spotify data as JSON"):
        with st.spinner("Fetching your data..."):
            data = fetch_data(sp)

        json_str = json.dumps(data, indent=2)
        st.download_button(
            "💾 Download JSON",
            json_str,
            file_name="spotify_data.json",
            mime="application/json",
        )
        st.json(data, expanded=False)

if __name__ == "__main__":
    main()
