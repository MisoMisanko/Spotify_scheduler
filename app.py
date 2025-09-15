# app.py
import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

# -----------------------------
# Config
# -----------------------------
SCOPES = "user-top-read playlist-read-private user-library-read"

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")  # <-- make sure this is in secrets.toml

st.set_page_config(page_title="Spotify + Last.fm Enrichment", page_icon="🎧", layout="centered")
st.title("🎧 Spotify + Last.fm Data Viewer")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()
if not LASTFM_API_KEY:
    st.warning("⚠️ Missing Last.fm API key. Add LASTFM_API_KEY to your secrets if you want enrichment.")

# -----------------------------
# Auth helpers
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
            token_info = auth_manager.get_access_token(code)
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
# Last.fm
# -----------------------------
def lastfm_request(method: str, **params):
    """Generic Last.fm API request helper."""
    if not LASTFM_API_KEY:
        return {}
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "method": method,
        **params
    }
    r = requests.get(base_url, params=query, timeout=10)
    r.raise_for_status()
    return r.json()

def enrich_with_lastfm(artist_name: str) -> dict:
    """Fetch Last.fm info for an artist name."""
    if not LASTFM_API_KEY:
        return {}
    try:
        res = lastfm_request("artist.getInfo", artist=artist_name)
        if "artist" in res:
            last_artist = res["artist"]
            playcount = int(last_artist.get("stats", {}).get("playcount", 0))
            tags = [t["name"] for t in last_artist.get("tags", {}).get("tag", [])]
            return {
                "lfm_playcount": playcount,
                "lfm_tags": tags[:5]
            }
    except Exception as e:
        return {"lfm_error": str(e)}
    return {}

# -----------------------------
# Data Fetch
# -----------------------------
def fetch_enriched_data(sp):
    # Top tracks & top artists
    top_tracks = sp.current_user_top_tracks(limit=20, time_range="medium_term")["items"]
    top_artists = sp.current_user_top_artists(limit=20, time_range="medium_term")["items"]

    # Collect unique artist IDs
    artist_ids = {a["id"] for a in top_artists if a.get("id")}
    for t in top_tracks:
        for a in t.get("artists", []):
            if a.get("id"):
                artist_ids.add(a["id"])
    artist_ids = list(artist_ids)

    # Get details from Spotify
    artist_details = {}
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            results = sp.artists(batch)["artists"]
            for a in results:
                artist_info = {
                    "name": a.get("name"),
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity", 0),
                    "followers": a.get("followers", {}).get("total", 0)
                }
                # Add Last.fm enrichment
                lfm_info = enrich_with_lastfm(a.get("name"))
                artist_info.update(lfm_info)
                artist_details[a["id"]] = artist_info
        except Exception:
            continue

    return top_tracks, top_artists, artist_details

# -----------------------------
# Display Helpers
# -----------------------------
def clean_track(t):
    return {
        "name": t.get("name"),
        "artists": [a["name"] for a in t.get("artists", [])],
        "album": t.get("album", {}).get("name"),
        "release_date": t.get("album", {}).get("release_date")
    }

# -----------------------------
# Main
# -----------------------------
sp = ensure_spotify_client()

if st.button("🔎 Pull my Spotify data"):
    with st.spinner("Fetching your Spotify + Last.fm data..."):
        top_tracks, top_artists, artist_details = fetch_enriched_data(sp)

    # Top tracks
    st.subheader("🎵 Top Tracks")
    for t in top_tracks[:10]:
        track_info = clean_track(t)
        st.write(f"- {track_info['name']} — {', '.join(track_info['artists'])} "
                 f"(Album: {track_info['album']}, Release: {track_info['release_date']})")

    # Enriched artists
    st.subheader("🎤 Enriched Artists")
    for a in list(artist_details.values())[:10]:
        genres = ", ".join(a["genres"]) if a["genres"] else "N/A"
        tags = ", ".join(a.get("lfm_tags", [])) if a.get("lfm_tags") else "N/A"
        st.write(f"- {a['name']} "
                 f"(Genres: {genres} | Popularity: {a['popularity']} | Followers: {a['followers']} | "
                 f"Last.fm playcount: {a.get('lfm_playcount','?')} | Tags: {tags})")

    with st.expander("📦 Raw JSON"):
        st.json(artist_details)
