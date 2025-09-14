# app.py
import os
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SCOPES = (
    "user-top-read playlist-read-private user-library-read "
    "user-follow-read user-read-recently-played"
)

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

st.set_page_config(page_title="Festival Personality App (AI)", page_icon="🎧", layout="centered")
st.title("🎧 Festival Personality App — Heuristics + AI")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()

def get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log((c/total) + 1e-12, 2) for c in counter.values())

def normalize(x, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def year_from_release_date(rd: str):
    if not rd:
        return None
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

    params = get_query_params()
    if "code" in params:
        code = params["code"][0] if isinstance(params["code"], list) else params["code"]
        try:
            token_info = _exchange_code(auth_manager, code)
            st.session_state["token_info"] = token_info
            clear_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Could not complete Spotify login: {e}")
            clear_query_params()
            st.stop()

    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()

def fetch_user_data(sp: spotipy.Spotify) -> dict:
    if "spotify_data" in st.session_state:
        return st.session_state["spotify_data"]

    data = {}

    # Only medium term data for simplicity
    data["top_tracks"]  = sp.current_user_top_tracks(limit=30, time_range="medium_term").get("items", [])
    data["top_artists"] = sp.current_user_top_artists(limit=30, time_range="medium_term").get("items", [])

    # Collect artist IDs
    artist_ids = set()
    artist_ids.update([a.get("id") for a in data["top_artists"]])
    artist_ids.update(safe_artist_ids(data["top_tracks"]))
    artist_ids = [a for a in artist_ids if a]

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

    st.session_state["spotify_data"] = data
    return data


def compute_signals(sp: spotipy.Spotify, data: dict) -> dict:
    signals = {}

    # Track IDs from top tracks
    track_ids = [t["id"] for t in data["top_tracks"] if t.get("id")]
    features = []
    for b in batch(track_ids, 50):
        try:
            feats = sp.audio_features(b)
            if feats: features.extend([f for f in feats if f])
        except Exception:
            features = []
            break

    if features:
        signals["avg_energy"]  = float(np.mean([f["energy"] for f in features if f.get("energy") is not None]))
        signals["avg_valence"] = float(np.mean([f["valence"] for f in features if f.get("valence") is not None]))
    else:
        signals["avg_energy"]  = 0.5
        signals["avg_valence"] = 0.5
        st.warning("No audio features found; using defaults.", icon="⚠️")

    # Artist genres & stats
    artist_genres, artist_pops, artist_follows = [], [], []
    for det in data["artist_details"].values():
        artist_pops.append(det.get("popularity", 0))
        artist_follows.append(det.get("followers", 0))
        artist_genres.extend(det.get("genres", []))

    genre_counter = Counter([g.lower() for g in artist_genres])
    signals["genre_entropy"]   = shannon_entropy(genre_counter)
    signals["genre_diversity"] = len(genre_counter)/sum(genre_counter.values()) if genre_counter else 0.0

    signals["avg_artist_popularity"] = float(np.mean(artist_pops)) if artist_pops else 0.0
    signals["long_tail_share"]       = (sum(1 for f in artist_follows if f < 100_000) / len(artist_follows)) if artist_follows else 0.0

    buckets = genre_bucket_counts(list(genre_counter.keys()))
    total_bucket = max(1, buckets.get("total", 0))
    signals["mellow_ratio"]     = buckets.get("mellow", 0) / total_bucket
    signals["aggressive_ratio"] = buckets.get("aggressive", 0) / total_bucket
    signals["dancepop_ratio"]   = buckets.get("dancepop", 0) / total_bucket

    return signals


def compute_traits(signals: dict) -> dict[str, float]:
    openness = (
        0.4 * normalize(signals["genre_entropy"], 5.5, 7.5) +
        0.3 * normalize(signals["genre_diversity"], 0.4, 0.8) +
        0.2 * normalize(signals["novelty_index"], 0.2, 0.8) +
        0.1 * normalize(signals["long_track_share"], 0.0, 0.3)
    )
    extraversion = (
        0.4 * normalize(signals["avg_artist_popularity"], 30.0, 80.0) +
        0.3 * normalize(signals["recency_share"], 0.1, 0.7) +
        0.3 * normalize(signals["dancepop_ratio"], 0.0, 0.4)
    )
    agreeableness = (
        0.6 * normalize(signals["mellow_ratio"] - signals["aggressive_ratio"], -0.3, 0.3) +
        0.4 * (1.0 - normalize(signals["long_tail_share"], 0.1, 0.6))
    )
    conscientiousness = (
        0.5 * normalize(signals["stability_index"], 0.1, 0.8) +
        0.3 * (1.0 - normalize(signals["long_track_share"], 0.0, 0.2)) +
        0.2 * normalize(signals["collab_playlist_ratio"], 0.0, 0.4)
    )
    energy     = normalize(signals.get("avg_energy", 0.5), 0.3, 0.9)
    positivity = normalize(signals.get("avg_valence", 0.5), 0.2, 0.8)

    return {
        "Openness": openness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Conscientiousness": conscientiousness,
        "Energy": energy,
        "Positivity": positivity
    }

FEATURE_COLUMNS = [
    "genre_entropy","genre_diversity","novelty_index","long_track_share",
    "avg_artist_popularity","recency_share","dancepop_ratio",
    "mellow_ratio","aggressive_ratio","long_tail_share","stability_index",
    "collab_playlist_ratio","avg_energy","avg_valence"
]
TARGET_COLUMNS = ["Openness","Extraversion","Agreeableness","Conscientiousness"]

def build_feature_vector(signals: dict) -> np.ndarray:
    return np.array([signals.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=float)

def ensure_ai_model():
    return st.session_state.get("ai_model")

def train_ai_model(df: pd.DataFrame):
    missing_f = [c for c in FEATURE_COLUMNS if c not in df.columns]
    missing_t = [c for c in TARGET_COLUMNS  if c not in df.columns]
    if missing_f or missing_t:
        raise ValueError(f"Missing columns. Features: {missing_f}, Targets: {missing_t}")
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values
    model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    model.fit(X, y)
    st.session_state["ai_model"] = model
    return model

st.sidebar.header("🤖 AI Model")
st.sidebar.write("Upload a CSV with columns:")
st.sidebar.code(", ".join(FEATURE_COLUMNS + TARGET_COLUMNS), language="text")

uploaded = st.sidebar.file_uploader("Upload training CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        _ = train_ai_model(df)
        st.sidebar.success("Model trained and stored in session.")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

if st.sidebar.button("Clear trained model"):
    st.session_state.pop("ai_model", None)
    st.sidebar.info("Cleared.")

if st.sidebar.button("Reset cached Spotify data"):
    st.session_state.pop("spotify_data", None)
    st.sidebar.info("Cache cleared.")

sp = ensure_spotify_client()

if st.button("🔎 Analyze my Spotify"):
    with st.spinner("Fetching your Spotify data and analysing it..."):
        data    = fetch_user_data(sp)
        signals = compute_signals(sp, data)

        model = ensure_ai_model()
        if model is not None:
            X = build_feature_vector(signals).reshape(1, -1)
            preds = np.clip(model.predict(X).reshape(-1), 0.0, 1.0)
            traits = {k: float(v) for k, v in zip(TARGET_COLUMNS, preds)}
            traits["Energy"]     = normalize(signals.get("avg_energy", 0.5), 0.3, 0.9)
            traits["Positivity"] = normalize(signals.get("avg_valence", 0.5), 0.2, 0.8)
            mode_used = "AI model (Ridge)"
        else:
            traits = compute_traits(signals)
            mode_used = "Heuristic"

        mbti = classify_mbti(traits["Openness"], traits["Extraversion"], traits["Agreeableness"], traits["Conscientiousness"])

    st.caption(f"Mode: {mode_used}")
    cols = st.columns(len(traits))
    for i, (k, v) in enumerate(traits.items()):
        with cols[i]:
            st.metric(k, f"{v:.2f}")

    labels = list(traits.keys())
    values = list(traits.values())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2,0.4,0.6,0.8])
    ax.set_title("Personality radar", pad=20)
    st.pyplot(fig)

    st.markdown(
        f"""<div style='text-align:center; padding:20px; border-radius:10px;
        background-color:#f0f9f9; border:2px solid #00b3b3;'>
        <h2 style='color:#007777;'>MBTI (derived from traits)</h2>
        <h1 style='color:#00b3b3;'>{mbti}</h1></div>""",
        unsafe_allow_html=True
    )

    with st.expander("🔍 Raw signals"):
        st.json({k: round(float(v),3) for k, v in signals.items()})
