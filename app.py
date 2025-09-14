# app.py
import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SCOPES = "user-top-read playlist-read-private user-library-read"

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(page_title="Spotify Data Viewer", page_icon="🎧", layout="centered")
st.title("🎧 Spotify Data Viewer")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()


# -----------------------------
# Auth
# -----------------------------
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=None,
        open_browser=False,
        show_dialog=True
    )

def _exchange_code(auth_manager, code):
    try:
        return auth_manager.get_access_token(code, as_dict=True)
    except TypeError:
        token = auth_manager.get_access_token(code)
        if isinstance(token, dict):
            return token
        return {"access_token": token, "expires_at": None, "refresh_token": None, "scope": SCOPES}

def ensure_spotify_client() -> spotipy.Spotify:
    auth_manager = get_auth_manager()
    token_info = st.session_state.get("token_info")

    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    if token_info and auth_manager.is_token_expired(token_info):
        try:
            st.session_state["token_info"] = auth_manager.refresh_access_token(token_info["refresh_token"])
            return spotipy.Spotify(auth=st.session_state["token_info"]["access_token"])
        except Exception:
            st.session_state.pop("token_info", None)

    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0] if isinstance(params["code"], list) else params["code"]
        try:
            token_info = _exchange_code(auth_manager, code)
            st.session_state["token_info"] = token_info
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Could not complete Spotify login: {e}")
            st.experimental_set_query_params()
            st.stop()

    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def enrich_artists(sp, artists):
    """Fetch missing genres/popularity/followers for artists with incomplete data."""
    enriched = []
    for a in artists:
        if not a.get("genres") or not a.get("popularity"):
            try:
                full = sp.artist(a["id"])
                a["genres"] = full.get("genres", [])
                a["popularity"] = full.get("popularity", a.get("popularity"))
                a["followers"] = full.get("followers", {}).get("total", a.get("followers", 0))
            except Exception:
                pass
        enriched.append(a)
    return enriched

def clean_track(t):
    return {
        "name": t.get("name"),
        "artists": [a["name"] for a in t.get("artists", [])],
        "album": t.get("album", {}).get("name"),
        "release_date": t.get("album", {}).get("release_date")
    }

def clean_artist(a):
    return {
        "name": a.get("name"),
        "genres": a.get("genres", []),
        "popularity": a.get("popularity"),
        "followers": a.get("followers", {}).get("total", 0)
    }


# -----------------------------
# Main
# -----------------------------
sp = ensure_spotify_client()

if st.button("🔎 Pull my Spotify data"):
    with st.spinner("Fetching your Spotify data..."):
        top_tracks = sp.current_user_top_tracks(limit=20, time_range="medium_term")["items"]
        top_artists = sp.current_user_top_artists(limit=20, time_range="medium_term")["items"]

        # Enrich artist info
        top_artists = enrich_artists(sp, top_artists)

    st.subheader("🎵 Top Tracks (medium term)")
    for t in top_tracks:
        track_info = clean_track(t)
        st.write(f"- {track_info['name']} — {', '.join(track_info['artists'])} "
                 f"(Album: {track_info['album']}, Release: {track_info['release_date']})")

    st.subheader("🎤 Top Artists (medium term)")
    for a in top_artists:
        artist_info = clean_artist(a)
        genres = ", ".join(artist_info["genres"]) if artist_info["genres"] else "N/A"
        st.write(f"- {artist_info['name']} "
                 f"(Genres: {genres} | Popularity: {artist_info['popularity']} | Followers: {artist_info['followers']})")

    with st.expander("📦 Raw JSON"):
        st.json({"top_tracks": top_tracks, "top_artists": top_artists})
