import os
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
load_dotenv()  # this reads .env automatically

# -----------------------------
# Config & Auth
# -----------------------------
SCOPES = "user-top-read playlist-read-private user-library-read user-follow-read"
CLIENT_ID = os.environ["SPOTIPY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIPY_CLIENT_SECRET"]
REDIRECT_URI = os.environ["SPOTIPY_REDIRECT_URI"]

st.set_page_config(page_title="Festival Personality App", page_icon="🎧", layout="centered")
st.title("🎧 Festival Personality App")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPES,
    cache_path=".cache",
    open_browser=False,    # 👈 stops trying localhost
    show_dialog=True       # 👈 forces login each time (good for testing)
))


# -----------------------------
# Helpers
# -----------------------------
def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log((c/total) + 1e-12, 2) for c in counter.values())

def normalize(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def year_from_release_date(rd: str) -> int:
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

MELLOW_GENRES = {"acoustic", "folk", "singer-songwriter", "ambient", "chill", "lo-fi", "soft rock"}
AGGRESSIVE_GENRES = {"metal", "hardcore", "punk", "industrial"}
DANCE_POP_GENRES = {"dance pop", "edm", "house", "techno", "trance"}

def genre_bucket_counts(genres):
    buckets = defaultdict(int)
    for g in genres:
        lg = g.lower()
        if any(x in lg for x in MELLOW_GENRES):
            buckets["mellow"] += 1
        if any(x in lg for x in AGGRESSIVE_GENRES):
            buckets["aggressive"] += 1
        if any(x in lg for x in DANCE_POP_GENRES):
            buckets["dancepop"] += 1
        buckets["total"] += 1
    return buckets

# -----------------------------
# Data collection
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_user_data():
    data = {}
    time_ranges = ["short_term", "medium_term", "long_term"]

    # Top tracks & artists
    data["top_tracks"] = {tr: sp.current_user_top_tracks(limit=50, time_range=tr).get("items", []) for tr in time_ranges}
    data["top_artists"] = {tr: sp.current_user_top_artists(limit=50, time_range=tr).get("items", []) for tr in time_ranges}

    # Saved tracks (sample)
    saved = []
    results = sp.current_user_saved_tracks(limit=50)
    saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
    while results.get("next") and len(saved) < 200:
        results = sp.next(results)
        saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
    data["saved_tracks"] = saved[:200]

    # Playlists
    pls = sp.current_user_playlists(limit=20).get("items", [])
    data["playlists"] = pls

    # Artist details
    artist_ids = set()
    for tr in time_ranges:
        artist_ids.update([a.get("id") for a in data["top_artists"][tr]])
        artist_ids.update(safe_artist_ids(data["top_tracks"][tr]))
    artist_ids.update(safe_artist_ids(data["saved_tracks"]))
    artist_ids = [a for a in artist_ids if a]

    artist_details = {}
    for b in batch(artist_ids, 50):
        res = sp.artists(b).get("artists", [])
        for a in res:
            artist_details[a["id"]] = {
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0),
                "genres": a.get("genres", [])
            }
    data["artist_details"] = artist_details
    return data

# -----------------------------
# Run analysis
# -----------------------------
if st.button("🔎 Analyze my Spotify"):
    data = fetch_user_data()

    # All unique tracks
    all_tracks = []
    for v in data["top_tracks"].values():
        all_tracks.extend(v)
    all_tracks.extend(data["saved_tracks"])
    seen = set()
    unique_tracks = []
    for t in all_tracks:
        tid = t.get("id")
        if tid and tid not in seen:
            unique_tracks.append(t)
            seen.add(tid)

    # Artist stats
    artist_genres = []
    artist_pops = []
    artist_follows = []
    for aid, det in data["artist_details"].items():
        artist_pops.append(det["popularity"])
        artist_follows.append(det["followers"])
        artist_genres.extend(det["genres"])

    # Features
    genre_counter = Counter([g.lower() for g in artist_genres])
    genre_entropy = shannon_entropy(genre_counter)
    unique_genres = len(genre_counter)
    genre_diversity = unique_genres / max(1, sum(genre_counter.values()))

    avg_artist_pop = float(np.mean(artist_pops)) if artist_pops else 0.0
    niche_artist_share = sum(1 for p in artist_pops if p < 40) / len(artist_pops) if artist_pops else 0.0
    long_tail_share = sum(1 for f in artist_follows if f < 100_000) / len(artist_follows) if artist_follows else 0.0

    durations_min = [t["duration_ms"]/60000 for t in unique_tracks if t.get("duration_ms")]
    avg_duration = float(np.mean(durations_min)) if durations_min else 0.0
    long_track_share = sum(1 for d in durations_min if d >= 6.0) / len(durations_min) if durations_min else 0.0

    years = [year_from_release_date(t.get("album", {}).get("release_date")) for t in unique_tracks]
    years = [y for y in years if y]
    current_year = datetime.utcnow().year
    recency_share = sum(1 for y in years if (current_year - y) <= 2) / len(years) if years else 0.0

    short_ids = {a.get("id") for a in data["top_artists"]["short_term"]}
    long_ids = {a.get("id") for a in data["top_artists"]["long_term"]}
    stability_index = len(short_ids & long_ids) / max(1, len(short_ids | long_ids))
    novelty_index = 1 - stability_index

    playlists = data["playlists"]
    collab_count = sum(1 for p in playlists if p.get("collaborative"))
    collab_ratio = (collab_count / max(1, len(playlists))) if playlists else 0.0

    buckets = genre_bucket_counts(list(genre_counter.keys()))
    mellow_ratio = buckets.get("mellow", 0) / max(1, buckets.get("total", 0))
    aggressive_ratio = buckets.get("aggressive", 0) / max(1, buckets.get("total", 0))
    dancepop_ratio = buckets.get("dancepop", 0) / max(1, buckets.get("total", 0))

    # Personality scoring
    openness = (
        0.45 * normalize(genre_entropy, 0.0, 8.0) +
        0.25 * normalize(genre_diversity, 0.0, 0.7) +
        0.15 * normalize(long_track_share, 0.0, 0.5) +
        0.15 * normalize(novelty_index, 0.0, 1.0)
    )
    extraversion = (
        0.50 * normalize(avg_artist_pop, 20.0, 80.0) +
        0.25 * normalize(collab_ratio, 0.0, 0.5) +
        0.25 * normalize(recency_share, 0.0, 0.8)
    )
    agreeableness = (
        0.65 * normalize(mellow_ratio - aggressive_ratio, -0.5, 0.5) +
        0.35 * normalize(dancepop_ratio, 0.0, 0.5)
    )
    conscientiousness = (
        0.7 * normalize(stability_index, 0.0, 1.0) +
        0.3 * (1 - normalize(long_track_share, 0.0, 0.5))
    )

    traits = {
        "Openness": openness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Conscientiousness": conscientiousness
    }

    # MBTI
    def classify_mbti(o, e, a, c):
        ei = "E" if e >= 0.5 else "I"
        ns = "N" if o >= 0.5 else "S"
        ft = "F" if a >= 0.5 else "T"
        jp = "J" if c >= 0.5 else "P"
        return ei + ns + ft + jp

    mbti = classify_mbti(openness, extraversion, agreeableness, conscientiousness)

    # -----------------------------
    # UI OUTPUT
    # -----------------------------
    st.subheader("🎭 Personality scores")
    cols = st.columns(4)
    for i, (trait, val) in enumerate(traits.items()):
        with cols[i]:
            st.metric(trait, round(val, 2))

    # Radar chart
    labels = list(traits.keys())
    values = list(traits.values())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="teal", alpha=0.25)
    ax.plot(angles, values, color="teal", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2,0.4,0.6,0.8])
    st.pyplot(fig)

    # MBTI
    st.markdown(f"""
    <div style='text-align:center; padding:20px; border-radius:10px;
    background-color:#f0f9f9; border:2px solid #00b3b3;'>
    <h2 style='color:#007777;'>Your MBTI Type</h2>
    <h1 style='color:#00b3b3;'>{mbti}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Festival recs
    st.subheader("🎶 Festival Recommendations")
    recs = []
    if openness > 0.6:
        recs.append("🎨 Explore discovery zones & art installations")
    else:
        recs.append("🎤 Stick with headliners & familiar stages")

    if extraversion > 0.6:
        recs.append("💃 Join big dance tents & group activities")
    else:
        recs.append("🧘 Relax in acoustic/chill-out areas")

    if conscientiousness > 0.6:
        recs.append("📅 Plan workshops & structured schedule")
    else:
        recs.append("✨ Go with the flow & explore pop-up events")

    for r in recs:
        st.markdown(f"- {r}")

    # Raw signals
    with st.expander("🔍 See raw analysis signals"):
        st.json({
            "unique_genres": unique_genres,
            "genre_entropy": round(genre_entropy, 3),
            "avg_artist_popularity": round(avg_artist_pop, 2),
            "niche_artist_share": round(niche_artist_share, 3),
            "long_tail_share": round(long_tail_share, 3),
            "avg_duration_min": round(avg_duration, 2),
            "long_track_share": round(long_track_share, 3),
            "recency_share(last 2y)": round(recency_share, 3),
            "stability_index": round(stability_index, 3),
            "collab_playlist_ratio": round(collab_ratio, 3),
            "mellow_ratio": round(mellow_ratio, 3),
            "aggressive_ratio": round(aggressive_ratio, 3),
            "dancepop_ratio": round(dancepop_ratio, 3)
        })
