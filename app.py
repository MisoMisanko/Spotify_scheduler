# app.py
from __future__ import annotations
import os, time, math, json, traceback
from typing import Dict, List, Any, Iterator, Tuple, Set
from collections import defaultdict, Counter

import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =======================
# CONFIG & PAGE
# =======================
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
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify + Last.fm Personality", page_icon="🎧", layout="wide")
st.title("🎧 Spotify + Last.fm – Personality Profiler")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()
if not LASTFM_API_KEY:
    st.error("Missing Last.fm credentials. Set LASTFM_API_KEY.")
    st.stop()

# =======================
# AUTH
# =======================
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

def _exchange_code(auth_manager: SpotifyOAuth, code: str) -> Dict[str, Any]:
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

# =======================
# HELPERS
# =======================
def batch(iterable: List[Any], n: int = 50) -> Iterator[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# Simple in-memory caches so we don't hammer Last.fm
_lastfm_artist_cache: Dict[str, Dict[str, Any]] = {}
_lastfm_track_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

def lastfm_request(method: str, **params: Any) -> Dict[str, Any]:
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {"api_key": LASTFM_API_KEY, "format": "json", "method": method, **params}
    r = requests.get(base_url, params=query, timeout=10)
    r.raise_for_status()
    return r.json()

def get_lastfm_artist(name: str) -> Dict[str, Any]:
    key = name.lower().strip()
    if key in _lastfm_artist_cache:
        return _lastfm_artist_cache[key]
    try:
        data = lastfm_request("artist.getInfo", artist=name, autocorrect=1)
        _lastfm_artist_cache[key] = data
        time.sleep(0.08)  # polite delay
        return data
    except Exception:
        _lastfm_artist_cache[key] = {}
        return {}

def get_lastfm_track(artist: str, track: str) -> Dict[str, Any]:
    key = (artist.lower().strip(), track.lower().strip())
    if key in _lastfm_track_cache:
        return _lastfm_track_cache[key]
    try:
        data = lastfm_request("track.getInfo", artist=artist, track=track, autocorrect=1)
        _lastfm_track_cache[key] = data
        time.sleep(0.08)
        return data
    except Exception:
        _lastfm_track_cache[key] = {}
        return {}

# =======================
# GENRE/TAG NORMALIZATION
# =======================
# Canonical buckets — blend of Rentfrow & Gosling dimensions + pragmatic groups
GENRE_BUCKETS = {
    "reflective_complex": {"classical", "baroque", "romantic era", "chamber", "symphony", "jazz", "bebop", "cool jazz",
                           "post-bop", "fusion jazz", "ambient", "post-rock", "neo-classical", "instrumental", "piano"},
    "intense_rebellious": {"metal", "heavy metal", "black metal", "death metal", "thrash", "hardcore", "punk", "emo",
                           "screamo", "grindcore", "industrial"},
    "upbeat_conventional": {"country", "adult contemporary", "soft rock", "oldies", "singer-songwriter", "acoustic pop",
                            "easy listening"},
    "energetic_rhythmic": {"edm", "electronic", "dance", "house", "techno", "trance", "drum and bass", "dubstep",
                           "garage", "electro"},
    # Practical groupings for modern listening
    "hip_hop": {"hip hop", "rap", "trap", "grime"},
    "rnb_soul": {"r&b", "rnb", "soul", "neo-soul"},
    "pop": {"pop", "k-pop", "j-pop", "europop", "indie pop", "synthpop", "dance pop"},
    "indie_alt": {"indie", "indie rock", "alternative", "alt-rock", "shoegaze", "lo-fi"},
    "folk_acoustic": {"folk", "indie folk", "acoustic", "americana"},
    "latin": {"latin", "reggaeton", "salsa", "bachata", "bossa nova", "cumbia"},
    "jazz": {"jazz", "swing", "latin jazz"},
    "classical": {"classical", "opera", "orchestral", "choral"},
}

def normalize_token(token: str) -> str:
    return token.lower().replace("-", " ").replace("/", " ").strip()

def map_to_buckets(all_tokens: List[str]) -> Dict[str, int]:
    """Map mixed Spotify genres and Last.fm tags to canonical buckets."""
    counts = Counter()
    tokens = [normalize_token(t) for t in all_tokens if t]
    for t in tokens:
        matched = False
        for bucket, vocab in GENRE_BUCKETS.items():
            # match on prefix containment
            if any(t == v or t.startswith(v) or v in t for v in vocab):
                counts[bucket] += 1
                matched = True
        # optionally: leave unmatched tokens for diversity calc
    return counts

# =======================
# DATA COLLECTION
# =======================
@st.cache_data(show_spinner=False, persist=False)
def fetch_user_profile(sp: spotipy.Spotify) -> Dict[str, Any]:
    try:
        return sp.current_user() or {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, persist=False)
def fetch_all_saved_tracks(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    out = []
    try:
        results = sp.current_user_saved_tracks(limit=50)
    except Exception:
        results = None
    while results:
        out.extend([it.get("track") for it in results.get("items", []) if it.get("track")])
        if results.get("next"):
            try: results = sp.next(results)
            except Exception: break
        else: break
    return out

@st.cache_data(show_spinner=False, persist=False)
def fetch_all_playlists_and_tracks(sp: spotipy.Spotify) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    playlists, playlist_tracks = [], {}
    try:
        results = sp.current_user_playlists(limit=50)
    except Exception:
        results = None
    while results:
        items = results.get("items", []) if results else []
        playlists.extend(items)
        for pl in items:
            pid = pl.get("id")
            if not pid: continue
            tracks = []
            try:
                res = sp.playlist_tracks(pid, limit=100)
            except Exception:
                res = None
            while res:
                tracks.extend([t.get("track") for t in res.get("items", []) if t.get("track")])
                if res.get("next"):
                    try: res = sp.next(res)
                    except Exception: break
                else: break
            playlist_tracks[pid] = tracks
        if results and results.get("next"):
            try: results = sp.next(results)
            except Exception: break
        else: break
    return playlists, playlist_tracks

@st.cache_data(show_spinner=False, persist=False)
def fetch_all_followed_artists(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    out = []
    try:
        results = sp.current_user_followed_artists(limit=50)
    except Exception:
        results = None
    while results:
        artists_data = results.get("artists", {}) if results else {}
        out.extend(artists_data.get("items", []))
        next_url = artists_data.get("next")
        if next_url:
            try:
                results = sp._get(next_url)
            except Exception:
                break
        else:
            break
    return out

@st.cache_data(show_spinner=False, persist=False)
def fetch_top(sp: spotipy.Spotify) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    time_ranges = ["short_term", "medium_term", "long_term"]
    top_tracks, top_artists = {}, {}
    for tr in time_ranges:
        try:
            top_tracks[tr] = sp.current_user_top_tracks(limit=50, time_range=tr).get("items", [])
        except Exception:
            top_tracks[tr] = []
        try:
            top_artists[tr] = sp.current_user_top_artists(limit=50, time_range=tr).get("items", [])
        except Exception:
            top_artists[tr] = []
    return top_tracks, top_artists

@st.cache_data(show_spinner=False, persist=False)
def fetch_recent(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    try:
        return sp.current_user_recently_played(limit=50).get("items", [])
    except Exception:
        return []

def collect_unique_track_ids(data: Dict[str, Any]) -> List[str]:
    track_ids = []
    for v in data["top_tracks"].values():
        track_ids.extend([t.get("id") for t in v if t and t.get("id")])
    track_ids.extend([t.get("id") for t in data["saved_tracks"] if t and t.get("id")])
    for tracks in data["playlist_tracks"].values():
        track_ids.extend([t.get("id") for t in tracks if t and t.get("id")])
    track_ids.extend([it.get("track", {}).get("id") for it in data["recent"] if it.get("track")])
    # de-dupe and keep strings only
    return list({tid for tid in track_ids if isinstance(tid, str) and tid})

def collect_unique_artist_ids(data: Dict[str, Any]) -> List[str]:
    a_ids = []
    for v in data["top_artists"].values():
        a_ids.extend([a.get("id") for a in v if a and a.get("id")])
    for tracks in data["playlist_tracks"].values():
        for t in tracks:
            for a in t.get("artists", []):
                if a.get("id"): a_ids.append(a["id"])
    for t in data["saved_tracks"]:
        for a in t.get("artists", []):
            if a.get("id"): a_ids.append(a["id"])
    for it in data["recent"]:
        for a in it.get("track", {}).get("artists", []):
            if a.get("id"): a_ids.append(a["id"])
    return list({aid for aid in a_ids if isinstance(aid, str) and aid})

def fetch_track_details_and_features(sp: spotipy.Spotify, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Robust: filters bad IDs; falls back to per-track; skips on errors."""
    details = {}
    total = len(track_ids)
    prog = st.empty()
    for i, chunk in enumerate(batch(track_ids, 50), start=1):
        valid_ids = [tid for tid in chunk if tid and isinstance(tid, str)]
        prog.text(f"Fetching track metadata/features: batch {i}/{math.ceil(total/50)}")
        if not valid_ids:
            continue
        # tracks
        try:
            tracks = sp.tracks(valid_ids).get("tracks", [])
        except Exception:
            tracks = [None] * len(valid_ids)
        # audio features with robust fallback
        try:
            feats = sp.audio_features(valid_ids) or []
        except Exception:
            feats = []
            for tid in valid_ids:
                try:
                    fl = sp.audio_features([tid])
                    feats.append(fl[0] if fl and fl[0] else None)
                except Exception:
                    feats.append(None)
        # align by position
        for tid, t_info, f_info in zip(valid_ids, tracks, feats):
            if not t_info:
                continue
            artists = [a.get("name") for a in t_info.get("artists", []) if a.get("name")]
            meta = {
                "name": t_info.get("name"),
                "artists": artists,
                "album": t_info.get("album", {}).get("name"),
                "release_date": t_info.get("album", {}).get("release_date"),
                "popularity": t_info.get("popularity", 0),
                "duration_ms": t_info.get("duration_ms"),
            }
            audio_fields = [
                "danceability","energy","loudness","speechiness","acousticness",
                "instrumentalness","liveness","valence","tempo","key","mode","time_signature"
            ]
            for f in audio_fields:
                meta[f] = f_info.get(f) if f_info else None

            # Last.fm track enrichment (safe)
            lfm = {}
            try:
                if artists and meta.get("name"):
                    tr = get_lastfm_track(artists[0], meta["name"])
                    tblock = tr.get("track", {})
                    playcount = tblock.get("playcount")
                    if playcount is not None: lfm["lfm_playcount"] = int(playcount)
                    tags = tblock.get("toptags", {}).get("tag", [])
                    if isinstance(tags, list):
                        lfm["lfm_tags"] = [t.get("name") for t in tags[:10] if t.get("name")]
            except Exception:
                pass

            meta.update(lfm)
            details[tid] = meta
    prog.empty()
    return details

def fetch_artist_details(sp: spotipy.Spotify, artist_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    details = {}
    total = len(artist_ids)
    prog = st.empty()
    for i, chunk in enumerate(batch(artist_ids, 50), start=1):
        prog.text(f"Fetching artist metadata: batch {i}/{math.ceil(total/50)}")
        try:
            arts = sp.artists(chunk).get("artists", [])
        except Exception:
            arts = []
        for a in arts:
            aid = a.get("id")
            if not aid: continue
            meta = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0)
            }
            # Last.fm artist enrichment
            try:
                la = get_lastfm_artist(a.get("name") or "")
                ablock = la.get("artist", {})
                stats = ablock.get("stats", {})
                pc = stats.get("playcount")
                if pc is not None: meta["lfm_playcount"] = int(pc)
                tags = ablock.get("tags", {}).get("tag", [])
                if isinstance(tags, list):
                    meta["lfm_tags"] = [t.get("name") for t in tags[:10] if t.get("name")]
            except Exception:
                pass
            details[aid] = meta
    prog.empty()
    return details

# =======================
# PERSONALITY HEURISTICS
# =======================
def safe_mean(vals: List[float]) -> float:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return sum(nums)/len(nums) if nums else 0.0

def normalize01(x: float, lo: float, hi: float) -> float:
    if x is None: return 0.0
    if hi == lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def compute_personality(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    tracks = list(data.get("track_details", {}).values())
    artists = list(data.get("artist_details", {}).values())

    # Aggregate audio features
    feats = defaultdict(list)
    for t in tracks:
        for k in ["danceability","energy","valence","acousticness","instrumentalness","speechiness","liveness","loudness","tempo"]:
            v = t.get(k)
            if isinstance(v, (int, float)):
                feats[k].append(v)
    # Means
    avg = {k: safe_mean(v) for k, v in feats.items()}
    # Normalized loudness/tempo
    loud_norm = normalize01(avg.get("loudness"), -60.0, 0.0)
    tempo_norm = normalize01(avg.get("tempo"), 0.0, 200.0)

    # Genre & tag pooling
    tokens: List[str] = []
    for a in artists:
        tokens.extend(a.get("genres", []))
        tokens.extend(a.get("lfm_tags", []))  # artist tags
    for t in tracks:
        tokens.extend(t.get("lfm_tags", []))  # track tags

    bucket_counts = map_to_buckets(tokens)
    total_bucket_hits = sum(bucket_counts.values()) or 1
    bucket_share = {k: v/total_bucket_hits for k, v in bucket_counts.items()}
    genre_diversity = len(set([normalize_token(x) for x in tokens if x])) / (len(tokens) or 1)

    # Mainstreamness: average popularity of tracks (0..100)
    pop_vals = [t.get("popularity", 0) for t in tracks if isinstance(t.get("popularity", 0), (int, float))]
    mainstream = (sum(pop_vals) / len(pop_vals) / 100.0) if pop_vals else 0.0

    # Heuristics aligned with literature:
    # - Extraversion: ↑ danceability, ↑ valence, ↑ energetic/rhythmic & pop share
    extraversion = (
        avg.get("danceability", 0) * 0.4
        + avg.get("valence", 0) * 0.3
        + (bucket_share.get("energetic_rhythmic", 0) + bucket_share.get("pop", 0)) * 0.3
    )

    # - Conscientiousness: ↓ energy, ↓ tempo, ↓ intense/rebellious, ↑ mainstreamness
    conscientiousness = (
        (1 - avg.get("energy", 0)) * 0.35
        + (1 - tempo_norm) * 0.2
        + (1 - bucket_share.get("intense_rebellious", 0)) * 0.25
        + mainstream * 0.2
    )

    # - Agreeableness: ↑ acousticness, ↓ energy, ↓ speechiness, ↓ live; ↑ folk/acoustic, singer-songwriter
    agreeableness = (
        avg.get("acousticness", 0) * 0.35
        + (1 - avg.get("energy", 0)) * 0.2
        + (1 - avg.get("speechiness", 0)) * 0.15
        + (bucket_share.get("folk_acoustic", 0) + bucket_share.get("upbeat_conventional", 0)) * 0.3
    )

    # - Openness: ↑ genre diversity, ↑ acousticness & instrumentalness, ↓ loudness & energy, ↑ reflective/complex
    openness = (
        genre_diversity * 0.3
        + avg.get("acousticness", 0) * 0.15
        + avg.get("instrumentalness", 0) * 0.15
        + (1 - loud_norm) * 0.15
        + (1 - avg.get("energy", 0)) * 0.1
        + bucket_share.get("reflective_complex", 0) * 0.15
    )

    # - Neuroticism: ↓ danceability, ↓ valence, ↑ energy, ↑ intense/rebellious
    neuroticism = (
        (1 - avg.get("danceability", 0)) * 0.3
        + (1 - avg.get("valence", 0)) * 0.3
        + avg.get("energy", 0) * 0.2
        + bucket_share.get("intense_rebellious", 0) * 0.2
    )

    scores = {
        "Extraversion": max(0.0, min(1.0, extraversion)),
        "Conscientiousness": max(0.0, min(1.0, conscientiousness)),
        "Agreeableness": max(0.0, min(1.0, agreeableness)),
        "Openness": max(0.0, min(1.0, openness)),
        "Neuroticism": max(0.0, min(1.0, neuroticism)),
    }
    debug = {
        "audio_means": avg,
        "loudness_norm": loud_norm,
        "tempo_norm": tempo_norm,
        "bucket_counts": dict(bucket_counts),
        "bucket_share": bucket_share,
        "genre_diversity": genre_diversity,
        "mainstreamness": mainstream,
    }
    return scores, debug

# =======================
# UI
# =======================
def main():
    sp = ensure_spotify_client()

    st.sidebar.header("Run Options")
    do_lastfm = st.sidebar.checkbox("Include Last.fm enrichment", value=True)
    st.sidebar.caption("Disable to speed up during testing.")

    if st.button("🔎 Pull everything + infer personality"):
        with st.spinner("Fetching your Spotify profile..."):
            user = fetch_user_profile(sp)
            st.success(f"Hello, {user.get('display_name') or 'Spotify user'}! 👋")

        with st.spinner("Pulling Spotify data (this may take a bit)..."):
            top_tracks, top_artists = fetch_top(sp)
            saved_tracks = fetch_all_saved_tracks(sp)
            playlists, playlist_tracks = fetch_all_playlists_and_tracks(sp)
            followed_artists = fetch_all_followed_artists(sp)
            recent = fetch_recent(sp)

        data = {
            "top_tracks": top_tracks,
            "top_artists": top_artists,
            "saved_tracks": saved_tracks,
            "playlists": playlists,
            "playlist_tracks": playlist_tracks,
            "followed_artists": followed_artists,
            "recent": recent,
        }

        t_ids = collect_unique_track_ids(data)
        a_ids = collect_unique_artist_ids(data)

        st.info(f"Unique tracks: {len(t_ids)}  |  Unique artists: {len(a_ids)}")

        with st.spinner("Enriching tracks with audio features + Last.fm..."):
            track_details = fetch_track_details_and_features(sp, t_ids if do_lastfm else t_ids)
        data["track_details"] = track_details

        with st.spinner("Enriching artists (Spotify + Last.fm)..."):
            artist_details = fetch_artist_details(sp, a_ids if do_lastfm else a_ids)
        data["artist_details"] = artist_details

        st.subheader("🎼 Enriched Tracks (first 12)")
        for t in list(track_details.values())[:12]:
            tags = ", ".join(t.get("lfm_tags", []) or []) or "—"
            st.write(
                f"- {t.get('name')} — {', '.join(t.get('artists', []) or [])} "
                f"(Album: {t.get('album')}, Pop: {t.get('popularity')}, "
                f"Dance: {t.get('danceability')}, Energy: {t.get('energy')}, "
                f"Valence: {t.get('valence')}, Tempo: {t.get('tempo')}) | Tags: {tags}"
            )

        st.subheader("🎤 Enriched Artists (first 12)")
        for a in list(artist_details.values())[:12]:
            genres = ", ".join(a.get("genres", []) or []) or "—"
            tags = ", ".join(a.get("lfm_tags", []) or []) or "—"
            st.write(
                f"- {a.get('name')} | Genres: {genres} | Pop: {a.get('popularity')} | "
                f"Followers: {a.get('followers')} | Last.fm playcount: {a.get('lfm_playcount','—')} | Tags: {tags}"
            )

        with st.spinner("Inferring personality (Big Five)…"):
            scores, debug = compute_personality(data)

        st.subheader("🧠 Personality (Heuristic, 0–1)")
        dom = max(scores.items(), key=lambda kv: kv[1])
        st.write(f"**Dominant trait:** {dom[0]} ({dom[1]:.2f})")
        for k, v in scores.items():
            st.write(f"- {k}: {v:.2f}")

        with st.expander("🔬 Debug signals (for transparency)"):
            st.json(debug)

        with st.expander("📦 Raw JSON (core)"):
            compact = {
                "counts": {
                    "unique_tracks": len(t_ids),
                    "unique_artists": len(a_ids),
                    "saved_tracks": len(saved_tracks),
                    "playlists": len(playlists),
                    "recent": len(recent),
                    "followed_artists": len(followed_artists),
                },
                "scores": scores,
            }
            st.json(compact)

if __name__ == "__main__":
    main()
