import os
import json
import time
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

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

st.set_page_config(page_title="Spotify Full Data Exporter", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Full Data Exporter")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials")
    st.stop()

# -----------------------------------------------------------------------------
# Auth
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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fetch_all(pager_func, key: str, limit: int = 50):
    """Generic paginator for Spotify endpoints that support .next()"""
    out = []
    try:
        results = pager_func(limit=limit)
    except Exception:
        return []
    while results:
        items = results.get("items", [])
        out.extend(items)
        if results.get("next"):
            try:
                results = sp.next(results)
            except Exception:
                break
        else:
            break
    return out

# -----------------------------------------------------------------------------
# Full Data Collection
# -----------------------------------------------------------------------------
def fetch_all_data(sp):
    data = {}

    # User profile
    try:
        data["user"] = sp.current_user()
    except Exception:
        data["user"] = {}

    # Top tracks & artists
    data["top_tracks"] = {}
    data["top_artists"] = {}
    for r in ["short_term", "medium_term", "long_term"]:
        try:
            data["top_tracks"][r] = sp.current_user_top_tracks(limit=50, time_range=r).get("items", [])
        except Exception:
            data["top_tracks"][r] = []
        try:
            data["top_artists"][r] = sp.current_user_top_artists(limit=50, time_range=r).get("items", [])
        except Exception:
            data["top_artists"][r] = []

    # Saved tracks (entire library)
    data["saved_tracks"] = fetch_all(sp.current_user_saved_tracks, "items")

    # Playlists and playlist tracks
    playlists = fetch_all(sp.current_user_playlists, "items")
    data["playlists"] = playlists
    playlist_tracks = {}
    for pl in playlists:
        pid = pl.get("id")
        if not pid:
            continue
        tracks = fetch_all(lambda limit: sp.playlist_tracks(pid, limit=limit), "items", limit=100)
        playlist_tracks[pid] = tracks
        time.sleep(0.05)  # gentle rate limiting
    data["playlist_tracks"] = playlist_tracks

    # Followed artists
    followed = []
    try:
        results = sp.current_user_followed_artists(limit=50)
    except Exception:
        results = None
    while results:
        artists_block = results.get("artists", {})
        followed.extend(artists_block.get("items", []))
        if artists_block.get("next"):
            try:
                results = sp.next(artists_block)
            except Exception:
                break
        else:
            break
    data["followed_artists"] = followed

    # Recently played
    try:
        data["recent"] = sp.current_user_recently_played(limit=50).get("items", [])
    except Exception:
        data["recent"] = []

    return data

# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------
def main():
    global sp
    sp = ensure_spotify_client()
    st.success("Spotify authenticated ✅")

    if st.button("📦 Export ALL Spotify data as JSON"):
        with st.spinner("Fetching everything... (this may take several minutes)"):
            data = fetch_all_data(sp)

        json_str = json.dumps(data, indent=2)
        st.download_button(
            "💾 Download JSON",
            json_str,
            file_name="spotify_full_data.json",
            mime="application/json",
        )
        st.info("✅ Export complete. Preview below:")
        st.json({k: len(v) if isinstance(v, list) else "ok" for k, v in data.items()})

if __name__ == "__main__":
    main()
