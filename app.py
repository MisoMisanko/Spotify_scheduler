"""
Streamlit app: Spotify + Last.fm Personality Profiler (fast demo version)
-------------------------------------------------------------------------

- Pulls top tracks (short/medium/long term), top artists, and recent plays.
- Enriches only artists with Last.fm tags/playcounts (fast).
- Skips track-level Last.fm enrichment (the main bottleneck).
- Infers Big Five traits from genres, tags, diversity, and mainstreamness.

Research basis:
- Extraversion → energetic, rhythmic, pop/dance, high valence.
- Conscientiousness → mainstream, structured, low rebellious/intense.
- Agreeableness → acoustic/folk, mellow, low energy.
- Openness → diverse, reflective/complex, acoustic/instrumental.
- Neuroticism → intense/rebellious, low valence/danceability.
"""

from __future__ import annotations
import os, time, math
from typing import Dict, List, Any, Iterator, Tuple
from collections import Counter

import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCOPES = (
    "user-top-read "
    "user-read-recently-played"
)
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify + Last.fm Personality (Fast)", page_icon="🎧", layout="wide")
st.title("🎧 Spotify + Last.fm — Personality Profiler (Fast Demo)")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials in environment variables.")
    st.stop()
if not LASTFM_API_KEY:
    st.error("Missing Last.fm credentials (LASTFM_API_KEY).")
    st.stop()

# -----------------------------------------------------------------------------
# Spotify auth
# -----------------------------------------------------------------------------
def get_auth_manager() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=None,
        open_browser=False,
        show_dialog=True,
    )

def _exchange_code(auth_manager: SpotifyOAuth, code: str):
    try:
        return auth_manager.get_access_token(code, as_dict=True)
    except TypeError:
        token = auth_manager.get_access_token(code)
        return token if isinstance(token, dict) else {"access_token": token}

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
            st.error(f"Spotify login failed: {e}")
            st.experimental_set_query_params()
            st.stop()

    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def batch(iterable: List[Any], n: int = 50):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# Last.fm cache
_lastfm_artist_cache: Dict[str, Dict[str, Any]] = {}
def lastfm_request(method: str, **params: Any) -> Dict[str, Any]:
    r = requests.get("http://ws.audioscrobbler.com/2.0/", params={
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "method": method,
        **params
    }, timeout=10)
    r.raise_for_status()
    return r.json()

def get_lastfm_artist(name: str) -> Dict[str, Any]:
    key = (name or "").strip().lower()
    if not key: return {}
    if key in _lastfm_artist_cache:
        return _lastfm_artist_cache[key]
    try:
        data = lastfm_request("artist.getInfo", artist=name, autocorrect=1)
    except Exception:
        data = {}
    _lastfm_artist_cache[key] = data
    time.sleep(0.05)
    return data

# Genre buckets
GENRE_BUCKETS = {
    "reflective_complex": {"classical","jazz","ambient","instrumental","piano","orchestral"},
    "intense_rebellious": {"metal","punk","emo","hardcore","industrial"},
    "upbeat_conventional": {"country","soft rock","singer songwriter","acoustic pop"},
    "energetic_rhythmic": {"edm","electronic","dance","house","techno","trance","dnb","dubstep"},
    "hip_hop": {"hip hop","rap","trap","grime"},
    "rnb_soul": {"r&b","soul","neo soul"},
    "pop": {"pop","k pop","j pop","synthpop","dance pop"},
    "indie_alt": {"indie","alternative","shoegaze","lo fi"},
    "folk_acoustic": {"folk","acoustic","americana"},
    "latin": {"latin","reggaeton","salsa","cumbia"},
}
def norm_token(s: str) -> str:
    return (s or "").lower().replace("-", " ").replace("/", " ").strip()

def map_tokens_to_buckets(tokens: List[str]) -> Counter:
    counts = Counter()
    for raw in tokens:
        t = norm_token(raw)
        for bucket, vocab in GENRE_BUCKETS.items():
            if any(t == v or t.startswith(v) or v in t for v in vocab):
                counts[bucket] += 1
    return counts

# -----------------------------------------------------------------------------
# Spotify pulls
# -----------------------------------------------------------------------------
def fetch_user_profile(sp): 
    try: return sp.current_user() or {}
    except: return {}

def fetch_top(sp):
    time_ranges = ["short_term", "medium_term", "long_term"]
    return {
        "tracks": {tr: sp.current_user_top_tracks(limit=50, time_range=tr).get("items", []) for tr in time_ranges},
        "artists": {tr: sp.current_user_top_artists(limit=50, time_range=tr).get("items", []) for tr in time_ranges}
    }

def fetch_recent(sp):
    try: return sp.current_user_recently_played(limit=50).get("items", [])
    except: return []

# -----------------------------------------------------------------------------
# Enrichment
# -----------------------------------------------------------------------------
def fetch_artist_details(sp, artist_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    details = {}
    for chunk in batch(artist_ids, 50):
        try: arts = sp.artists(chunk).get("artists", [])
        except: arts = []
        for a in arts:
            aid = a.get("id"); 
            if not aid: continue
            meta = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0),
            }
            try:
                la = get_lastfm_artist(a.get("name") or "")
                ablock = la.get("artist", {})
                stats = ablock.get("stats", {})
                if stats.get("playcount"): meta["lfm_playcount"] = int(stats["playcount"])
                tags = ablock.get("tags", {}).get("tag", [])
                if isinstance(tags, list): meta["lfm_tags"] = [t.get("name") for t in tags[:10] if t.get("name")]
            except: pass
            details[aid] = meta
    return details

# -----------------------------------------------------------------------------
# Personality
# -----------------------------------------------------------------------------
def safe_mean(vals: List[float]) -> float:
    nums = [v for v in vals if isinstance(v, (int,float))]
    return sum(nums)/len(nums) if nums else 0.0

def compute_personality(data: Dict[str, Any]):
    tracks = [t for tr in data["top"]["tracks"].values() for t in tr]
    artists = [a for ar in data["top"]["artists"].values() for a in ar]
    recents = [it.get("track") for it in data["recent"] if it.get("track")]
    tracks.extend([r for r in recents if r])

    tokens = []
    for a in data["artist_details"].values():
        tokens.extend(a.get("genres", []))
        tokens.extend(a.get("lfm_tags", []))
    bucket_counts = map_tokens_to_buckets(tokens)
    total_hits = sum(bucket_counts.values()) or 1
    bucket_share = {k:v/total_hits for k,v in bucket_counts.items()}
    genre_diversity = len({norm_token(x) for x in tokens if x})/(len(tokens) or 1)
    pops = [t.get("popularity") for t in tracks if isinstance(t.get("popularity"), (int,float))]
    mainstreamness = (sum(pops)/len(pops)/100.0) if pops else 0.0

    extraversion = (bucket_share.get("energetic_rhythmic",0)+bucket_share.get("pop",0))*0.7
    conscientiousness = (1-bucket_share.get("intense_rebellious",0))*0.5 + mainstreamness*0.5
    agreeableness = (bucket_share.get("folk_acoustic",0)+bucket_share.get("upbeat_conventional",0))*0.8
    openness = genre_diversity*0.5 + bucket_share.get("reflective_complex",0)*0.5
    neuroticism = bucket_share.get("intense_rebellious",0)*0.7 + (1-mainstreamness)*0.3

    scores = {
        "Openness": openness,
        "Conscientiousness": conscientiousness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Neuroticism": neuroticism,
    }
    return scores, {"bucket_counts":dict(bucket_counts), "bucket_share":bucket_share,
                    "genre_diversity":genre_diversity,"mainstreamness":mainstreamness}

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    sp = ensure_spotify_client()
    if st.button("🔎 Pull data and infer personality"):
        with st.spinner("Fetching Spotify data…"):
            user = fetch_user_profile(sp)
            top = fetch_top(sp); recent = fetch_recent(sp)
        st.success(f"Hello, {user.get('display_name') or 'Spotify user'}! 👋")

        artist_ids = list({a.get("id") for tr in top["artists"].values() for a in tr if a.get("id")})
        with st.spinner("Enriching artists via Last.fm…"):
            artist_details = fetch_artist_details(sp, artist_ids)

        data = {"top":top, "recent":recent, "artist_details":artist_details}
        with st.spinner("Inferring personality…"):
            scores, debug = compute_personality(data)

        st.subheader("🧠 Personality (scores 0–1)")
        dom = max(scores.items(), key=lambda kv: kv[1])
        st.write(f"**Dominant trait:** {dom[0]} ({dom[1]:.2f})")
        for k,v in scores.items():
            st.write(f"- {k}: {v:.2f}")

        with st.expander("🔬 Debug info"): st.json(debug)

if __name__=="__main__":
    main()
