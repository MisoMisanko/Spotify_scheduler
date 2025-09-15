# app.py
import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------
# Config
# -----------------------------
SCOPES = "user-top-read playlist-read-private user-library-read user-follow-read user-read-recently-played"

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify + Last.fm Data Viewer", page_icon="🎧", layout="centered")
st.title("🎧 Spotify + Last.fm Data Viewer")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()
if not LASTFM_API_KEY:
    st.error("Missing Last.fm credentials. Set LASTFM_API_KEY.")
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
def lastfm_request(method: str, **params):
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

def batch(iterable, n=50):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Data fetch + enrichment
# -----------------------------
def fetch_full_user_data(sp):
    data = {}

    # Top tracks/artists across ranges
    time_ranges = ["short_term", "medium_term", "long_term"]
    data["top_tracks"] = {tr: sp.current_user_top_tracks(limit=50, time_range=tr).get("items", []) for tr in time_ranges}
    data["top_artists"] = {tr: sp.current_user_top_artists(limit=50, time_range=tr).get("items", []) for tr in time_ranges}

    # Saved tracks (up to 200)
    saved = []
    results = sp.current_user_saved_tracks(limit=50)
    while results and len(saved) < 200:
        saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
        if results.get("next") and len(saved) < 200:
            results = sp.next(results)
        else:
            break
    data["saved_tracks"] = saved

    # Playlists (first 50)
    try:
        data["playlists"] = sp.current_user_playlists(limit=50).get("items", [])
    except Exception:
        data["playlists"] = []

    # Recently played
    try:
        data["recent"] = sp.current_user_recently_played(limit=50).get("items", [])
    except Exception:
        data["recent"] = []

    # ------------------------
    # Enrich tracks
    # ------------------------
    track_ids = set()
    for v in data["top_tracks"].values():
        track_ids.update([t["id"] for t in v if t.get("id")])
    track_ids.update([t["id"] for t in saved if t.get("id")])
    track_ids = list(track_ids)

    track_details = {}
    for b in batch(track_ids, 50):
        try:
            tracks = sp.tracks(b).get("tracks", [])
        except Exception:
            tracks = []
        try:
            feats = sp.audio_features(b) or []
        except Exception:
            feats = [None] * len(b)

        for tid, t, f in zip(b, tracks, feats):
            if not t:
                continue
            track_details[tid] = {
                "name": t.get("name"),
                "artists": [a["name"] for a in t.get("artists", [])],
                "album": t.get("album", {}).get("name"),
                "release_date": t.get("album", {}).get("release_date"),
                "popularity": t.get("popularity", 0),
                "duration_ms": t.get("duration_ms"),
                "energy": f.get("energy") if f else None,
                "valence": f.get("valence") if f else None,
                "danceability": f.get("danceability") if f else None,
                "tempo": f.get("tempo") if f else None,
            }
    data["track_details"] = track_details

    # ------------------------
    # Enrich artists
    # ------------------------
    artist_ids = set()
    for v in data["top_artists"].values():
        artist_ids.update([a.get("id") for a in v if a.get("id")])
    for t in track_details.values():
        for name in t["artists"]:
            # Skip: Spotify API needs IDs not names, but keep coverage
            pass
    artist_ids = [a for a in artist_ids if a]

    artist_details = {}
    for b in batch(artist_ids, 50):
        try:
            res = sp.artists(b).get("artists", [])
        except Exception:
            res = []
        for a in res:
            details = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0)
            }
            # Enrich with Last.fm
            try:
                lfm = lastfm_request("artist.getInfo", artist=a.get("name"))
                if "artist" in lfm:
                    stats = lfm["artist"].get("stats", {})
                    details["lfm_playcount"] = int(stats.get("playcount", 0))
                    tags = [t["name"] for t in lfm["artist"].get("tags", {}).get("tag", [])]
                    details["lfm_tags"] = tags[:10]
            except Exception:
                pass
            artist_details[a["id"]] = details
    data["artist_details"] = artist_details

    return data

# -----------------------------
# Main
# -----------------------------
sp = ensure_spotify_client()

if st.button("🔎 Pull my Spotify + Last.fm data"):
    with st.spinner("Fetching your Spotify + Last.fm data..."):
        data = fetch_full_user_data(sp)

    # Tracks
    st.subheader("🎼 Enriched Tracks (first 10)")
    for t in list(data["track_details"].values())[:10]:
        st.write(
            f"- {t['name']} — {', '.join(t['artists'])} "
            f"(Album: {t['album']}, Release: {t['release_date']}, "
            f"Energy: {t['energy']}, Valence: {t['valence']}, "
            f"Danceability: {t['danceability']}, Tempo: {t['tempo']})"
        )

    # Artists
    st.subheader("🎤 Enriched Artists (first 10)")
    for a in list(data["artist_details"].values())[:10]:
        genres = ", ".join(a.get("genres", [])) or "N/A"
        tags = ", ".join(a.get("lfm_tags", [])) if "lfm_tags" in a else "N/A"
        st.write(
            f"- {a['name']} | Genres: {genres} | Popularity: {a['popularity']} | Followers: {a['followers']} | "
            f"Last.fm playcount: {a.get('lfm_playcount', 'N/A')} | Tags: {tags}"
        )

    with st.expander("📦 Raw JSON"):
        st.json(data)
