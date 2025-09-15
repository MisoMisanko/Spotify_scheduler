# app.py
from __future__ import annotations

import os, time, math, concurrent.futures
from typing import Dict, List, Any, Iterator, Tuple
from collections import Counter, defaultdict

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
# CONFIG (tuned for ~5 min)
# =======================
SCOPES = (
    "user-top-read "
    "playlist-read-private "
    "user-library-read "
    "user-follow-read "
    "user-read-recently-played"
)

# Turbo caps (adjust if you want slower/faster)
MAX_SAVED_TRACKS_SAMPLE = 300
MAX_PLAYLISTS_SAMPLE = 20
MAX_TRACKS_PER_PLAYLIST_SAMPLE = 50
MAX_FOLLOWED_ARTISTS_SAMPLE = 200
MAX_LASTFM_ARTISTS = 100
MAX_LASTFM_TRACKS = 150
LASTFM_MAX_WORKERS = 8           # parallel threads for Last.fm
LASTFM_POLITE_DELAY = 0.0        # small delay per call (0–0.05); keep 0 for speed

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify + Last.fm Personality (Turbo)", page_icon="🎧", layout="wide")
st.title("🎧 Spotify + Last.fm — Personality Profiler (Turbo)")

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
        if isinstance(token, dict): return token
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

# Last.fm caches
_lfm_artist_cache: Dict[str, Dict[str, Any]] = {}
_lfm_track_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

def lastfm_request(method: str, **params: Any) -> Dict[str, Any]:
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {"api_key": LASTFM_API_KEY, "format": "json", "method": method, **params}
    r = requests.get(base_url, params=query, timeout=10)
    r.raise_for_status()
    return r.json()

def lfm_artist(name: str) -> Dict[str, Any]:
    key = (name or "").strip().lower()
    if not key: return {}
    if key in _lfm_artist_cache: return _lfm_artist_cache[key]
    try: data = lastfm_request("artist.getInfo", artist=name, autocorrect=1)
    except Exception: data = {}
    _lfm_artist_cache[key] = data
    if LASTFM_POLITE_DELAY: time.sleep(LASTFM_POLITE_DELAY)
    return data

def lfm_track(artist: str, track: str) -> Dict[str, Any]:
    a, t = (artist or "").strip().lower(), (track or "").strip().lower()
    if not a or not t: return {}
    key = (a, t)
    if key in _lfm_track_cache: return _lfm_track_cache[key]
    try: data = lastfm_request("track.getInfo", artist=artist, track=track, autocorrect=1)
    except Exception: data = {}
    _lfm_track_cache[key] = data
    if LASTFM_POLITE_DELAY: time.sleep(LASTFM_POLITE_DELAY)
    return data

# Genre buckets (macro + pragmatic)
GENRE_BUCKETS = {
    "reflective_complex": {"classical","baroque","romantic era","chamber","symphony","opera","orchestral","choral",
                           "jazz","bebop","cool jazz","post-bop","swing","latin jazz","fusion jazz","ambient",
                           "post rock","post-rock","neo classical","neo-classical","instrumental","piano"},
    "intense_rebellious": {"metal","heavy metal","black metal","death metal","thrash","hardcore","punk","emo","screamo",
                           "grindcore","industrial","alt metal"},
    "upbeat_conventional": {"country","adult contemporary","soft rock","oldies","singer songwriter","singer-songwriter",
                            "easy listening","acoustic pop"},
    "energetic_rhythmic": {"edm","electronic","dance","house","techno","trance","drum and bass","drum & bass","dnb",
                           "dubstep","garage","electro","bassline"},
    "hip_hop": {"hip hop","rap","trap","grime"},
    "rnb_soul": {"r&b","rnb","soul","neo-soul","neo soul"},
    "pop": {"pop","k pop","k-pop","j pop","j-pop","europop","indie pop","synthpop","dance pop"},
    "indie_alt": {"indie","indie rock","alternative","alt rock","alt-rock","shoegaze","lo fi","lo-fi"},
    "folk_acoustic": {"folk","indie folk","acoustic","americana"},
    "latin": {"latin","reggaeton","salsa","bachata","bossa nova","cumbia"},
}
def norm_token(s: str) -> str:
    return (s or "").lower().replace("-", " ").replace("/", " ").strip()
def map_tokens_to_buckets(tokens: List[str]) -> Counter:
    counts = Counter()
    for raw in tokens:
        t = norm_token(raw)
        if not t: continue
        for bucket, vocab in GENRE_BUCKETS.items():
            if any(t == v or t.startswith(v) or v in t for v in vocab):
                counts[bucket] += 1
    return counts

# =======================
# SPOTIFY PULLS (sampled for speed)
# =======================
def fetch_user_profile(sp: spotipy.Spotify) -> Dict[str, Any]:
    try: return sp.current_user() or {}
    except Exception: return {}

def fetch_top(sp: spotipy.Spotify):
    time_ranges = ["short_term", "medium_term", "long_term"]
    top_tracks, top_artists = {}, {}
    for tr in time_ranges:
        try: top_tracks[tr] = sp.current_user_top_tracks(limit=50, time_range=tr).get("items", [])
        except Exception: top_tracks[tr] = []
        try: top_artists[tr] = sp.current_user_top_artists(limit=50, time_range=tr).get("items", [])
        except Exception: top_artists[tr] = []
    return top_tracks, top_artists

def fetch_saved_sample(sp: spotipy.Spotify, cap=MAX_SAVED_TRACKS_SAMPLE):
    out, pulled = [], 0
    try: results = sp.current_user_saved_tracks(limit=50)
    except Exception: results = None
    while results and pulled < cap:
        items = [it.get("track") for it in results.get("items", []) if it.get("track")]
        out.extend(items); pulled += len(items)
        if pulled >= cap: break
        if results.get("next"):
            try: results = sp.next(results)
            except Exception: break
        else: break
    return out[:cap]

def fetch_playlists_sample(sp: spotipy.Spotify, pl_cap=MAX_PLAYLISTS_SAMPLE, tr_cap=MAX_TRACKS_PER_PLAYLIST_SAMPLE):
    playlists, playlist_tracks = [], {}
    try: results = sp.current_user_playlists(limit=50)
    except Exception: results = None
    while results and len(playlists) < pl_cap:
        items = results.get("items", []) if results else []
        remain = pl_cap - len(playlists)
        playlists.extend(items[:remain])
        for pl in items[:remain]:
            pid = pl.get("id"); tracks = []
            if not pid: continue
            try: res = sp.playlist_tracks(pid, limit=min(100, tr_cap))
            except Exception: res = None
            pulled = 0
            while res and pulled < tr_cap:
                batch_items = [t.get("track") for t in res.get("items", []) if t.get("track")]
                tracks.extend(batch_items); pulled += len(batch_items)
                if pulled >= tr_cap: break
                if res.get("next"):
                    try: res = sp.next(res)
                    except Exception: break
                else: break
            playlist_tracks[pid] = tracks[:tr_cap]
        if results.get("next") and len(playlists) < pl_cap:
            try: results = sp.next(results)
            except Exception: break
        else: break
    return playlists, playlist_tracks

def fetch_recent(sp: spotipy.Spotify):
    try: return sp.current_user_recently_played(limit=50).get("items", [])
    except Exception: return []

def fetch_followed_sample(sp: spotipy.Spotify, cap=MAX_FOLLOWED_ARTISTS_SAMPLE):
    out = []
    try: results = sp.current_user_followed_artists(limit=50)
    except Exception: results = None
    while results and len(out) < cap:
        artists_data = results.get("artists", {}) if results else {}
        items = artists_data.get("items", [])
        out.extend(items[: max(0, cap - len(out))])
        next_url = artists_data.get("next")
        if next_url and len(out) < cap:
            try: results = sp._get(next_url)
            except Exception: break
        else: break
    return out[:cap]

# =======================
# ID COLLECTION
# =======================
def collect_unique_track_ids(data: Dict[str, Any]) -> List[str]:
    ids = []
    for v in data["top_tracks"].values():
        ids.extend([t.get("id") for t in v if t and t.get("id") and t.get("type") == "track"])
    ids.extend([t.get("id") for t in data["saved_tracks"] if t and t.get("id") and t.get("type") == "track"])
    for tracks in data["playlist_tracks"].values():
        ids.extend([t.get("id") for t in tracks if t and t.get("id") and t.get("type") == "track"])
    ids.extend([it.get("track", {}).get("id") for it in data["recent"] if it.get("track") and it["track"].get("type") == "track"])
    return list({tid for tid in ids if tid})

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
    # Followed sample too
    for a in data.get("followed_artists", []):
        if a.get("id"): a_ids.append(a["id"])
    return list({aid for aid in a_ids if aid})

# =======================
# ENRICHMENT
# =======================
def fetch_artist_details(sp: spotipy.Spotify, artist_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    details = {}
    for chunk in batch(artist_ids, 50):
        try: arts = sp.artists(chunk).get("artists", [])
        except Exception: arts = []
        for a in arts:
            aid = a.get("id")
            if not aid: continue
            details[aid] = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0),
            }
    return details

def fetch_track_metadata(sp: spotipy.Spotify, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    details = {}
    for chunk in batch(track_ids, 50):
        try: tracks = sp.tracks(chunk).get("tracks", [])
        except Exception: tracks = []
        for t in tracks:
            if not t or t.get("type") != "track": continue
            tid = t.get("id")
            if not tid: continue
            details[tid] = {
                "name": t.get("name"),
                "artists": [a.get("name") for a in t.get("artists", []) if a.get("name")],
                "album": t.get("album", {}).get("name"),
                "release_date": t.get("album", {}).get("release_date"),
                "popularity": t.get("popularity", 0),
            }
    return details

def try_audio_features(sp: spotipy.Spotify, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Attempt to fetch audio features (only if your app still has extended access).
    Robust: batch first; on error, retry one-by-one; skip failing tracks.
    """
    feats_by_id = {}
    for chunk in batch(track_ids, 50):
        try:
            feats = sp.audio_features(chunk) or []
            for tid, f in zip(chunk, feats):
                if f: feats_by_id[tid] = f
        except Exception:
            # fallback one-by-one
            for tid in chunk:
                try:
                    f = sp.audio_features([tid])
                    if f and f[0]: feats_by_id[tid] = f[0]
                except Exception:
                    pass
    return feats_by_id

# --------- Last.fm parallel enrichment (sampled) ----------
def enrich_artists_with_lastfm(artists: Dict[str, Dict[str, Any]], max_artists=MAX_LASTFM_ARTISTS):
    # pick most popular artists to enrich (diagnostic + fast)
    if not artists: return
    top = sorted(artists.items(), key=lambda kv: kv[1].get("popularity", 0), reverse=True)[:max_artists]
    names = [v.get("name") for _, v in top if v.get("name")]
    if not names: return

    progress = st.progress(0, text="Last.fm: enriching artists…")
    done, total = 0, len(names)

    def worker(name: str):
        data = lfm_artist(name)
        tags = data.get("artist", {}).get("tags", {}).get("tag", [])
        stats = data.get("artist", {}).get("stats", {})
        return name, [t.get("name") for t in tags[:10] if t.get("name")], stats.get("playcount")

    with concurrent.futures.ThreadPoolExecutor(max_workers=LASTFM_MAX_WORKERS) as ex:
        futures = {ex.submit(worker, n): n for n in names}
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            progress.progress(min(done/total, 1.0))
            try:
                name, tags, playcount = fut.result()
                # write back into artist dict
                for aid, meta in artists.items():
                    if meta.get("name") == name:
                        if tags: meta["lfm_tags"] = tags
                        if playcount is not None: meta["lfm_playcount"] = int(playcount)
                        break
            except Exception:
                pass
    progress.empty()

def enrich_tracks_with_lastfm(tracks: Dict[str, Dict[str, Any]], max_tracks=MAX_LASTFM_TRACKS):
    # pick tracks by popularity to enrich
    if not tracks: return
    top = sorted(tracks.items(), key=lambda kv: kv[1].get("popularity", 0), reverse=True)[:max_tracks]
    pairs = [(v.get("artists", [None])[0], v.get("name")) for _, v in top if v.get("artists") and v.get("name")]

    progress = st.progress(0, text="Last.fm: enriching tracks…")
    done, total = 0, len(pairs)

    def worker(artist: str, title: str):
        data = lfm_track(artist, title)
        tblock = data.get("track", {})
        tags = tblock.get("toptags", {}).get("tag", [])
        playcount = tblock.get("playcount")
        return (artist, title), [t.get("name") for t in tags[:10] if t.get("name")], playcount

    with concurrent.futures.ThreadPoolExecutor(max_workers=LASTFM_MAX_WORKERS) as ex:
        futures = {ex.submit(worker, a, t): (a, t) for (a, t) in pairs}
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            progress.progress(min(done/total, 1.0))
            try:
                (a, t), tags, playcount = fut.result()
                # write back into tracks (first match)
                for tid, meta in tracks.items():
                    if (meta.get("artists") and meta["artists"][0] == a) and meta.get("name") == t:
                        if tags: meta["lfm_tags"] = tags
                        if playcount is not None: meta["lfm_playcount"] = int(playcount)
                        break
            except Exception:
                pass
    progress.empty()

# =======================
# PERSONALITY
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

    # tokens
    tokens: List[str] = []
    for a in artists:
        tokens.extend(a.get("genres", []) or [])
        tokens.extend(a.get("lfm_tags", []) or [])
    for t in tracks:
        tokens.extend(t.get("lfm_tags", []) or [])

    bucket_counts = map_tokens_to_buckets(tokens)
    total_hits = sum(bucket_counts.values()) or 1
    bucket_share = {k: v/total_hits for k, v in bucket_counts.items()}
    genre_diversity = len({norm_token(x) for x in tokens if x}) / (len(tokens) or 1)

    pops = [t.get("popularity") for t in tracks if isinstance(t.get("popularity"), (int, float))]
    mainstreamness = (sum(pops)/len(pops)/100.0) if pops else 0.0

    # Audio features (may be absent)
    feature_fields = ["danceability","energy","valence","acousticness","instrumentalness","speechiness","liveness","loudness","tempo"]
    coverage = {f: sum(1 for t in tracks if isinstance(t.get(f), (int, float))) for f in feature_fields}
    have_features = sum(coverage.values()) > 0
    avg = {}
    if have_features:
        for f in feature_fields: avg[f] = safe_mean([t.get(f) for t in tracks if isinstance(t.get(f), (int, float))])
        loud_norm = normalize01(avg.get("loudness"), -60.0, 0.0)
        tempo_norm = normalize01(avg.get("tempo"), 0.0, 200.0)
    else:
        loud_norm = tempo_norm = 0.0

    # Heuristic blend (works without features; uses them if present)
    extraversion = (
        (avg.get("danceability", 0)*0.40 + avg.get("valence", 0)*0.20 if have_features else 0) +
        (bucket_share.get("energetic_rhythmic", 0) + bucket_share.get("pop", 0)) * 0.40
    )
    conscientiousness = (
        ((1-avg.get("energy",0))*0.20 + (1-tempo_norm)*0.10 if have_features else 0) +
        (1-bucket_share.get("intense_rebellious",0))*0.45 + mainstreamness*0.25
    )
    agreeableness = (
        (avg.get("acousticness",0)*0.25 + (1-avg.get("energy",0))*0.10 + (1-avg.get("speechiness",0))*0.05 if have_features else 0) +
        (bucket_share.get("folk_acoustic",0) + bucket_share.get("upbeat_conventional",0))*0.60
    )
    openness = (
        genre_diversity*0.40 +
        ((avg.get("acousticness",0)+avg.get("instrumentalness",0))*0.20 + (1-loud_norm)*0.10 + (1-avg.get("energy",0))*0.05 if have_features else 0) +
        bucket_share.get("reflective_complex",0)*0.25
    )
    neuroticism = (
        ((1-avg.get("danceability",0))*0.25 + (1-avg.get("valence",0))*0.25 + avg.get("energy",0)*0.10 if have_features else 0.30) +
        bucket_share.get("intense_rebellious",0)*0.40
    )

    scores = {
        "Openness": max(0, min(1, openness)),
        "Conscientiousness": max(0, min(1, conscientiousness)),
        "Extraversion": max(0, min(1, extraversion)),
        "Agreeableness": max(0, min(1, agreeableness)),
        "Neuroticism": max(0, min(1, neuroticism)),
    }
    debug = {
        "bucket_counts": dict(bucket_counts),
        "bucket_share": bucket_share,
        "genre_diversity": genre_diversity,
        "mainstreamness": mainstreamness,
        "feature_coverage": coverage,
        "feature_means": avg if have_features else "no_features",
    }
    return scores, debug

# =======================
# UI
# =======================
def main():
    sp = ensure_spotify_client()

    st.sidebar.header("Run mode")
    include_features = st.sidebar.checkbox("Attempt Spotify audio features (if your app has extended access)", value=False)
    include_lastfm = st.sidebar.checkbox("Enrich with Last.fm (recommended)", value=True)
    st.sidebar.caption(
        """For speed, this run samples your library (top items, first 300 saved songs,
        first 20 playlists × 50 tracks, first 200 followed artists). You can
        raise the caps in the source to trade speed for coverage."""
    )

    if st.button("🔎 Pull sampled data + infer personality"):
        with st.spinner("Fetching profile…"):
            user = fetch_user_profile(sp)
        st.success(f"Hello, {user.get('display_name') or 'Spotify user'}! 👋")

        with st.spinner("Pulling sampled Spotify datasets…"):
            top_tracks, top_artists = fetch_top(sp)
            saved_tracks = fetch_saved_sample(sp)
            playlists, playlist_tracks = fetch_playlists_sample(sp)
            followed_artists = fetch_followed_sample(sp)
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
        st.info(f"Sampled: {len(t_ids)} unique tracks | {len(a_ids)} unique artists")

        with st.spinner("Fetching artist + track metadata…"):
            artist_details = fetch_artist_details(sp, a_ids)
            track_details = fetch_track_metadata(sp, t_ids)

        # Optional: Audio features (only if your app still has access)
        if include_features:
            with st.spinner("Attempting Spotify audio features…"):
                feats = try_audio_features(sp, list(track_details.keys()))
            # merge
            for tid, f in feats.items():
                if tid in track_details:
                    for k in ["danceability","energy","valence","acousticness","instrumentalness","speechiness","liveness","loudness","tempo","key","mode","time_signature"]:
                        track_details[tid][k] = f.get(k)

        # Last.fm enrichment on top-N items (parallel)
        if include_lastfm:
            enrich_artists_with_lastfm(artist_details, max_artists=MAX_LASTFM_ARTISTS)
            enrich_tracks_with_lastfm(track_details, max_tracks=MAX_LASTFM_TRACKS)

        data["artist_details"] = artist_details
        data["track_details"] = track_details

        # Show a small sample
        st.subheader("🎼 Enriched Tracks (sample of 10)")
        for t in list(track_details.values())[:10]:
            tags = ", ".join(t.get("lfm_tags", []) or []) or "—"
            st.write(f"- {t.get('name')} — {', '.join(t.get('artists', []) or [])} (Pop {t.get('popularity')}) | Tags: {tags}")

        st.subheader("🎤 Enriched Artists (sample of 10)")
        for a in list(artist_details.values())[:10]:
            genres = ", ".join(a.get("genres", []) or []) or "—"
            tags = ", ".join(a.get("lfm_tags", []) or []) or "—"
            st.write(
                f"- {a.get('name')} | Genres: {genres} | Pop {a.get('popularity')} "
                f"| Followers: {a.get('followers')} | LFM: {a.get('lfm_playcount','—')} | Tags: {tags}"
            )

        with st.spinner("Inferring personality…"):
            scores, debug = compute_personality(data)

        st.subheader("🧠 Personality (0–1)")
        dom = max(scores.items(), key=lambda kv: kv[1])
        st.write(f"**Dominant trait:** {dom[0]} ({dom[1]:.2f})")
        for k, v in scores.items():
            st.write(f"- {k}: {v:.2f}")

        with st.expander("🔎 Signals & coverage"):
            st.json(debug)

        with st.expander("📦 Dataset counts"):
            st.json({
                "saved_tracks": len(saved_tracks),
                "playlists": len(playlists),
                "playlist_tracks_total": sum(len(v) for v in playlist_tracks.values()),
                "recent_items": len(recent),
                "followed_artists_sample": len(followed_artists),
                "unique_track_ids": len(t_ids),
                "unique_artist_ids": len(a_ids),
            })

if __name__ == "__main__":
    main()
