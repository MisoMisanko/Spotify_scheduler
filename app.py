"""
Streamlit app for inferring Big Five personality from Spotify listening data.
Optimized for speed: enriches only top tracks (short/medium/long) and top artists,
with Last.fm enrichment on artists + a small set of tracks.
"""

import os
import time
import math
from typing import Dict, Any, List, Iterator, Tuple
from collections import Counter

import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCOPES = "user-top-read user-read-recently-played"
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify Personality Profiler", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Personality Profiler")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI and LASTFM_API_KEY):
    st.error("Missing credentials. Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, and LASTFM_API_KEY.")
    st.stop()

# -----------------------------------------------------------------------------
# Spotify Auth
# -----------------------------------------------------------------------------
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=None,
        open_browser=False,
    )

def ensure_spotify_client() -> spotipy.Spotify:
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
    st.info("Log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def batch(iterable: List[Any], n: int = 50) -> Iterator[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def lastfm_request(method: str, **params) -> Dict[str, Any]:
    url = "http://ws.audioscrobbler.com/2.0/"
    query = {"api_key": LASTFM_API_KEY, "format": "json", "method": method, **params}
    try:
        r = requests.get(url, params=query, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def get_lastfm_artist(name: str) -> Dict[str, Any]:
    try:
        return lastfm_request("artist.getInfo", artist=name, autocorrect=1)
    except Exception:
        return {}

def get_lastfm_track(artist: str, track: str) -> Dict[str, Any]:
    try:
        return lastfm_request("track.getInfo", artist=artist, track=track, autocorrect=1)
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# Data Fetch
# -----------------------------------------------------------------------------
def fetch_top(sp: spotipy.Spotify):
    ranges = ["short_term", "medium_term", "long_term"]
    top_tracks, top_artists = [], []
    for r in ranges:
        try:
            top_tracks.extend(sp.current_user_top_tracks(limit=30, time_range=r)["items"])
        except Exception:
            pass
        try:
            top_artists.extend(sp.current_user_top_artists(limit=30, time_range=r)["items"])
        except Exception:
            pass
    return top_tracks, top_artists

def fetch_recent(sp: spotipy.Spotify):
    try:
        return sp.current_user_recently_played(limit=30)["items"]
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Enrichment
# -----------------------------------------------------------------------------
def enrich_artists(sp, artists: List[Dict[str, Any]]):
    details = {}
    ids = [a["id"] for a in artists if a.get("id")]
    for chunk in batch(ids, 50):
        res = sp.artists(chunk).get("artists", [])
        for a in res:
            meta = {
                "name": a["name"],
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
            }
            lf = get_lastfm_artist(a["name"])
            tags = lf.get("artist", {}).get("tags", {}).get("tag", [])
            if isinstance(tags, list):
                meta["lfm_tags"] = [t["name"] for t in tags[:5] if "name" in t]
            details[a["id"]] = meta
            time.sleep(0.05)
    return details

def enrich_tracks(sp, tracks: List[Dict[str, Any]]):
    details = {}
    ids = [t["id"] for t in tracks if t.get("id")]
    for chunk in batch(ids, 50):
        res = sp.tracks(chunk).get("tracks", [])
        for t in res:
            artists = [a["name"] for a in t.get("artists", [])]
            meta = {
                "name": t["name"],
                "artists": artists,
                "popularity": t.get("popularity", 0),
            }
            lf = get_lastfm_track(artists[0], t["name"]) if artists else {}
            tags = lf.get("track", {}).get("toptags", {}).get("tag", [])
            if isinstance(tags, list):
                meta["lfm_tags"] = [tg["name"] for tg in tags[:5] if "name" in tg]
            details[t["id"]] = meta
            time.sleep(0.05)
    return details

# -----------------------------------------------------------------------------
# Personality
# -----------------------------------------------------------------------------
BUCKETS = {
    "extraversion": ["pop", "dance", "edm", "electro", "house"],
    "conscientiousness": ["classical", "soundtrack", "mainstream", "pop"],
    "agreeableness": ["folk", "acoustic", "singer-songwriter", "soul"],
    "openness": ["jazz", "world", "indie", "alt", "classical"],
    "neuroticism": ["metal", "punk", "emo", "hardcore"],
}

def compute_personality(artists, tracks):
    tokens = []
    for a in artists.values():
        tokens += a.get("genres", []) + a.get("lfm_tags", [])
    for t in tracks.values():
        tokens += t.get("lfm_tags", [])
    tokens = [t.lower() for t in tokens if t]

    raw = {trait: 0 for trait in BUCKETS}
    for trait, vocab in BUCKETS.items():
        raw[trait] = sum(1 for tok in tokens if any(v in tok for v in vocab))

    total = sum(raw.values()) or 1
    norm = {k: v/total for k, v in raw.items()}
    return norm

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sp = ensure_spotify_client()
    user = sp.current_user()
    st.success(f"Hello, {user.get('display_name', 'Spotify user')} 👋")

    with st.spinner("Fetching Spotify data..."):
        top_tracks, top_artists = fetch_top(sp)
        recent = fetch_recent(sp)

    st.info(f"Top tracks: {len(top_tracks)}, Top artists: {len(top_artists)}, Recent plays: {len(recent)}")

    with st.spinner("Enriching artists..."):
        artist_details = enrich_artists(sp, top_artists)
    with st.spinner("Enriching top tracks..."):
        track_details = enrich_tracks(sp, top_tracks)

    scores = compute_personality(artist_details, track_details)
    dominant = max(scores.items(), key=lambda x: x[1])

    st.subheader("🧠 Personality (scores 0–1)")
    st.write(f"**Dominant trait:** {dominant[0].title()} ({dominant[1]:.2f})")
    for k, v in scores.items():
        st.write(f"- {k.title()}: {v:.2f}")

if __name__ == "__main__":
    main()
