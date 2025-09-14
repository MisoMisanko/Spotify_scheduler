import os
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Optional: load from .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------
# Config
# -----------------------------
SCOPES = "user-top-read playlist-read-private user-library-read user-follow-read"

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(page_title="Festival Personality App", page_icon="🎧", layout="centered")
st.title("🎧 Festival Personality App")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error(
        "Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, "
        "SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI in your secrets/env."
    )
    st.stop()

# -----------------------------
# Small utilities
# -----------------------------
def get_query_params():
    """Compatible getter for Streamlit query params across versions."""
    try:
        return dict(st.query_params)  # Streamlit ≥1.31
    except Exception:
        return st.experimental_get_query_params()  # older

def clear_query_params():
    """Clear query params after token exchange to avoid loop/rerun issues."""
    try:
        st.query_params.clear()  # Streamlit ≥1.31
    except Exception:
        st.experimental_set_query_params()  # older (sets to empty)

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log((c/total) + 1e-12, 2) for c in counter.values())

def normalize(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def year_from_release_date(rd: str):
    if not rd: return None
    try:
        return int(rd[:4])
    except Exception:
        return None

def batch(iterable, n=50):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def safe_artist_ids(tracks):
    ids = []
    for t in tracks:
        artists = t.get("artists", [])
        if artists:
            aid = artists[0].get("id")
            if aid:
                ids.append(aid)
    return ids

MELLOW_GENRES = {"acoustic","folk","singer-songwriter","ambient","chill","lo-fi","soft rock"}
AGGRESSIVE_GENRES = {"metal","hardcore","punk","industrial"}
DANCE_POP_GENRES = {"dance pop","edm","house","techno","trance"}

def genre_bucket_counts(genres):
    buckets = defaultdict(int)
    for g in genres:
        lg = g.lower()
        if any(x in lg for x in MELLOW_GENRES):     buckets["mellow"] += 1
        if any(x in lg for x in AGGRESSIVE_GENRES): buckets["aggressive"] += 1
        if any(x in lg for x in DANCE_POP_GENRES):  buckets["dancepop"] += 1
        buckets["total"] += 1
    return buckets

def classify_mbti(o, e, a, c):
    ei = "E" if e >= 0.5 else "I"
    ns = "N" if o >= 0.5 else "S"
    ft = "F" if a >= 0.5 else "T"
    jp = "J" if c >= 0.5 else "P"
    return ei + ns + ft + jp

# -----------------------------
# OAuth & token management
# -----------------------------
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=".cache",     # ok for local dev; Cloud stores ephemeral copy
        open_browser=False,      # critical for Streamlit Cloud
        show_dialog=True         # helpful for incognito / fresh sessions
    )

def ensure_spotify_client() -> spotipy.Spotify:
    """
    Returns a ready-to-use spotipy.Spotify client.
    Handles first-time login, token exchange, refresh, and query param cleanup.
    """
    auth_manager = get_auth_manager()

    # 1) If we already have a token in session and it's valid → use it
    token_info = st.session_state.get("token_info")
    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    # 2) If we have a token but it's expired → refresh it
    if token_info and auth_manager.is_token_expired(token_info):
        try:
            st.session_state["token_info"] = auth_manager.refresh_access_token(
                token_info["refresh_token"]
            )
            return spotipy.Spotify(auth=st.session_state["token_info"]["access_token"])
        except Exception as e:
            # Fall back to login
            st.session_state.pop("token_info", None)

    # 3) If Spotify just redirected back with ?code=... → exchange it
    params = get_query_params()
    if "code" in params:
        code = params["code"][0] if isinstance(params["code"], list) else params["code"]
        try:
            token_info = auth_manager.get_access_token(code)
            st.session_state["token_info"] = token_info
            clear_query_params()   # remove ?code=... to prevent loop
            st.rerun()             # rerender with clean URL & token in state
        except Exception as e:
            st.error(f"Could not complete Spotify login: {e}")
            clear_query_params()
            st.stop()

    # 4) Otherwise → show login link and stop
    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

# -----------------------------
# Data collection
# -----------------------------
def fetch_user_data(sp: spotipy.Spotify) -> dict:
    data = {}
    time_ranges = ["short_term", "medium_term", "long_term"]

    # Keep limits modest for speed
    data["top_tracks"]  = {tr: sp.current_user_top_tracks(limit=20, time_range=tr).get("items", []) for tr in time_ranges}
    data["top_artists"] = {tr: sp.current_user_top_artists(limit=20, time_range=tr).get("items", []) for tr in time_ranges}

    # Saved tracks: cap at ~100 and avoid infinite paging
    saved = []
    try:
        results = sp.current_user_saved_tracks(limit=50)
        saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
        if results.get("next"):
            results = sp.next(results)
            saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
    except Exception:
        pass
    data["saved_tracks"] = saved[:100]

    # Playlists (first page only)
    try:
        data["playlists"] = sp.current_user_playlists(limit=20).get("items", [])
    except Exception:
        data["playlists"] = []

    # Unique artist IDs across all sources
    artist_ids = set()
    for tr in time_ranges:
        artist_ids.update([a.get("id") for a in data["top_artists"][tr]])
        artist_ids.update(safe_artist_ids(data["top_tracks"][tr]))
    artist_ids.update(safe_artist_ids(data["saved_tracks"]))
    artist_ids = [a for a in artist_ids if a]

    # Batch fetch artist details
    artist_details = {}
    for b in batch(artist_ids, 50):
        try:
            res = sp.artists(b).get("artists", [])
            for a in res:
                artist_details[a["id"]] = {
                    "popularity": a.get("popularity", 0),
                    "followers": a.get("followers", {}).get("total", 0),
                    "genres": a.get("genres", [])
                }
        except Exception:
            continue
    data["artist_details"] = artist_details
    return data

# -----------------------------
# Feature engineering
# -----------------------------
def compute_signals(data: dict) -> dict:
    signals = {}

    # Unique tracks
    unique_tracks = []
    seen = set()
    for v in data["top_tracks"].values():
        for t in v:
            tid = t.get("id")
            if tid and tid not in seen:
                unique_tracks.append(t); seen.add(tid)
    for t in data.get("saved_tracks", []):
        tid = t.get("id")
        if tid and tid not in seen:
            unique_tracks.append(t); seen.add(tid)

    # Fetch audio features for top tracks
    track_ids = [t["id"] for t in data["top_tracks"]["medium_term"] if t.get("id")]
    features = []
    for b in batch(track_ids, 50):
        try:
            features.extend(sp.audio_features(b))
        except Exception:
            pass

    if features:
        signals["avg_energy"] = float(np.mean([f["energy"] for f in features if f]))
        signals["avg_valence"] = float(np.mean([f["valence"] for f in features if f]))
    else:
        signals["avg_energy"] = 0.0
        signals["avg_valence"] = 0.0


    # Artist fields
    artist_genres, artist_pops, artist_follows = [], [], []
    for det in data["artist_details"].values():
        artist_pops.append(det.get("popularity", 0))
        artist_follows.append(det.get("followers", 0))
        artist_genres.extend(det.get("genres", []))

    # Genre entropy & diversity
    genre_counter = Counter([g.lower() for g in artist_genres])
    signals["genre_entropy"]   = shannon_entropy(genre_counter)
    unique_genres              = len(genre_counter)
    total_genres               = sum(genre_counter.values())
    signals["genre_diversity"] = unique_genres / total_genres if total_genres else 0.0

    # Popularity & long tail
    signals["avg_artist_popularity"] = float(np.mean(artist_pops)) if artist_pops else 0.0
    signals["niche_artist_share"]    = (sum(1 for p in artist_pops if p < 40) / len(artist_pops)) if artist_pops else 0.0
    signals["long_tail_share"]       = (sum(1 for f in artist_follows if f < 100_000) / len(artist_follows)) if artist_follows else 0.0

    # Duration
    durations_min = [t.get("duration_ms", 0)/60000 for t in unique_tracks if t.get("duration_ms")]
    signals["avg_duration_min"] = float(np.mean(durations_min)) if durations_min else 0.0
    signals["long_track_share"] = (sum(1 for d in durations_min if d >= 6.0)/len(durations_min)) if durations_min else 0.0

    # Recency (last 2y)
    years = []
    for t in unique_tracks:
        y = year_from_release_date(t.get("album", {}).get("release_date"))
        if y: years.append(y)
    current_year = datetime.utcnow().year
    signals["recency_share"] = (sum(1 for y in years if (current_year - y) <= 2)/len(years)) if years else 0.0

    # Stability vs novelty
    short_ids = {a.get("id") for a in data["top_artists"].get("short_term", [])}
    long_ids  = {a.get("id") for a in data["top_artists"].get("long_term", [])}
    inter     = len(short_ids & long_ids)
    union     = max(1, len(short_ids | long_ids))
    signals["stability_index"] = inter/union
    signals["novelty_index"]   = 1 - signals["stability_index"]

    # Collaborative playlists
    pls = data.get("playlists", [])
    collab_count = sum(1 for p in pls if p.get("collaborative"))
    signals["collab_playlist_ratio"] = (collab_count/len(pls)) if pls else 0.0

    # Genre buckets
    buckets = genre_bucket_counts(list(genre_counter.keys()))
    total_bucket = max(1, buckets.get("total", 0))
    signals["mellow_ratio"]     = buckets.get("mellow", 0) / total_bucket
    signals["aggressive_ratio"] = buckets.get("aggressive", 0) / total_bucket
    signals["dancepop_ratio"]   = buckets.get("dancepop", 0) / total_bucket

    return signals

def compute_traits(signals: dict) -> dict[str, float]:
    """
    Convert raw signals into more spread-out trait scores.
    Ranges have been tightened based on typical Spotify data.
    """

    # Openness: entropy + diversity + novelty + long tracks
    openness = (
        0.4 * normalize(signals["genre_entropy"], 5.5, 7.5) +
        0.3 * normalize(signals["genre_diversity"], 0.4, 0.8) +
        0.2 * normalize(signals["novelty_index"], 0.2, 0.8) +
        0.1 * normalize(signals["long_track_share"], 0.0, 0.3)
    )

    # Extraversion: popularity + recency + dancepop
    extraversion = (
        0.4 * normalize(signals["avg_artist_popularity"], 30.0, 80.0) +
        0.3 * normalize(signals["recency_share"], 0.1, 0.7) +
        0.3 * normalize(signals["dancepop_ratio"], 0.0, 0.4)
    )

    # Agreeableness: mellow vs aggressive, less smoothing
    agreeableness = (
        0.6 * normalize(signals["mellow_ratio"] - signals["aggressive_ratio"], -0.3, 0.3) +
        0.4 * (1.0 - normalize(signals["long_tail_share"], 0.1, 0.6))  # avoid extremes
    )

    # Conscientiousness: stability vs novelty, less collapse
    conscientiousness = (
        0.5 * normalize(signals["stability_index"], 0.1, 0.8) +
        0.3 * (1.0 - normalize(signals["long_track_share"], 0.0, 0.2)) +
        0.2 * normalize(signals["collab_playlist_ratio"], 0.0, 0.4)
    )

    # Energy (NEW): based on audio features
    energy = normalize(signals.get("avg_energy", 0.0), 0.3, 0.9)

    # Positivity (NEW): valence measure
    positivity = normalize(signals.get("avg_valence", 0.0), 0.2, 0.8)

    return {
        "Openness": openness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Conscientiousness": conscientiousness,
        "Energy": energy,
        "Positivity": positivity
    }



# -----------------------------
# Auth: get a working client
# -----------------------------
sp = ensure_spotify_client()  # will stop with login link if not authenticated

# -----------------------------
# UI
# -----------------------------
if st.button("🔎 Analyze my Spotify"):
    with st.spinner("Fetching your Spotify data and analysing it..."):
        data    = fetch_user_data(sp)
        signals = compute_signals(data)
        traits  = compute_traits(signals)
        mbti    = classify_mbti(traits["Openness"], traits["Extraversion"],
                                traits["Agreeableness"], traits["Conscientiousness"])

    st.subheader("🎭 Personality scores")
   # Create as many columns as there are traits
    cols = st.columns(len(traits))

    for i, (k, v) in enumerate(traits.items()):
        with cols[i]:
            st.metric(k, f"{v:.2f}")


    # Radar chart
    labels = list(traits.keys())
    values = list(traits.values())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values, linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([0.2,0.4,0.6,0.8]); ax.set_title("Personality radar", pad=20)
    st.pyplot(fig)

    st.markdown(
        f"""<div style='text-align:center; padding:20px; border-radius:10px;
        background-color:#f0f9f9; border:2px solid #00b3b3;'>
        <h2 style='color:#007777;'>Your MBTI Type</h2>
        <h1 style='color:#00b3b3;'>{mbti}</h1></div>""",
        unsafe_allow_html=True
    )



    with st.expander("🔍 See raw analysis signals"):
        st.json({k: round(v,3) for k, v in signals.items()})
