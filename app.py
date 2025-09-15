"""
Streamlit application for analysing a user's Spotify listening habits and
inferring their dominant Big Five personality trait.  The application
authenticates with Spotify using OAuth, retrieves as much data as
possible about the user's listening history (top tracks and artists,
saved tracks, playlists and their tracks, recently played tracks and
followed artists), enriches that data with metadata from the Last.fm
API and optionally with Spotify's audio features, and then computes
heuristic scores for each of the Big Five personality dimensions.

Important research foundations (used to derive the heuristics):

* **Extraversion** – People high in extraversion gravitate toward
  energetic, rhythmic and contemporary music.  Cross‑cultural studies
  show positive correlations between extraversion and upbeat genres as
  well as high danceability and valence【427880637911948†L63-L67】【934760148766067†L361-L369】.

* **Conscientiousness** – Conscientious listeners tend to avoid
  intense, aggressive styles【427880637911948†L63-L66】.  They prefer
  structured, mainstream music and show lower tolerance for high
  energy or unpredictable songs【934760148766067†L355-L369】.

* **Agreeableness** – Agreeable individuals enjoy mellow, warm and
  acoustic music【427880637911948†L63-L67】.  They often favour romantic
  and folk genres and show positive associations with acousticness and
  negative associations with energy and speechiness【934760148766067†L370-L390】.

* **Openness** – Openness is linked to appreciation for complex and
  diverse music, including classical, jazz and world genres【427880637911948†L63-L67】.
  On an audio level, openness correlates with acousticness and
  instrumentalness while being negatively associated with loudness,
  energy and tempo; open listeners also exhibit greater genre
  diversity【934760148766067†L336-L347】.

* **Neuroticism** – Neurotic listeners tend to prefer intense or
  emotionally charged music and show negative correlations with
  danceability and valence【934760148766067†L392-L396】.

Since November 27 2024 Spotify has restricted access to several Web
API endpoints, including **Audio Features** and **Audio Analysis**.
Only applications with pre‑existing extended access can query these
endpoints.  This application therefore disables audio feature
retrieval by default and falls back to genre and tag information
whenever audio features are unavailable【961739625275564†L57-L71】.
"""

from __future__ import annotations

import os
import time
import math
from typing import Dict, List, Any, Iterator, Tuple, Counter
from collections import Counter as CounterType

import streamlit as st
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -----------------------------------------------------------------------------
# Configuration and page setup
# -----------------------------------------------------------------------------

# Scopes needed for retrieving user data.  We request extensive read access
# (top tracks/artists, saved library, followed artists, recently played etc.).
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
st.title("🎧 Spotify + Last.fm — Personality Profiler")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error(
        "Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, "
        "SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI as environment variables."
    )
    st.stop()
if not LASTFM_API_KEY:
    st.error("Missing Last.fm credentials. Set LASTFM_API_KEY as an environment variable.")
    st.stop()

# -----------------------------------------------------------------------------
# Spotify authentication helpers
# -----------------------------------------------------------------------------

def get_auth_manager() -> SpotifyOAuth:
    """Configure a Spotify OAuth manager with the required scopes."""
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
    """Exchange an authorization code for an access token, handling legacy types."""
    try:
        return auth_manager.get_access_token(code, as_dict=True)
    except TypeError:
        token = auth_manager.get_access_token(code)
        if isinstance(token, dict):
            return token
        return {
            "access_token": token,
            "expires_at": None,
            "refresh_token": None,
            "scope": SCOPES,
        }

def ensure_spotify_client() -> spotipy.Spotify:
    """
    Ensure that a Spotipy client is available.  This function handles
    caching of the OAuth token in the Streamlit session state and
    automatically refreshes it when necessary.  If the user has not
    authenticated yet, the function presents a login link and stops
    execution until authentication completes.
    """
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

# -----------------------------------------------------------------------------
# Helper functions: batching, Last.fm requests and genre normalisation
# -----------------------------------------------------------------------------

def batch(iterable: List[Any], n: int = 50) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# Simple in-memory caches for Last.fm to avoid repeat network calls
_lastfm_artist_cache: Dict[str, Dict[str, Any]] = {}
_lastfm_track_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

def lastfm_request(method: str, **params: Any) -> Dict[str, Any]:
    """Low-level helper to call the Last.fm API with the given method and parameters."""
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "method": method,
        **params,
    }
    r = requests.get(base_url, params=query, timeout=10)
    r.raise_for_status()
    return r.json()

def get_lastfm_artist(name: str) -> Dict[str, Any]:
    """Retrieve artist information from Last.fm (with caching)."""
    key = (name or "").strip().lower()
    if not key:
        return {}
    if key in _lastfm_artist_cache:
        return _lastfm_artist_cache[key]
    try:
        data = lastfm_request("artist.getInfo", artist=name, autocorrect=1)
    except Exception:
        data = {}
    _lastfm_artist_cache[key] = data
    # polite delay to respect Last.fm rate limits
    time.sleep(0.05)
    return data

def get_lastfm_track(artist: str, track: str) -> Dict[str, Any]:
    """Retrieve track information from Last.fm (with caching)."""
    a = (artist or "").strip().lower()
    t = (track or "").strip().lower()
    if not a or not t:
        return {}
    key = (a, t)
    if key in _lastfm_track_cache:
        return _lastfm_track_cache[key]
    try:
        data = lastfm_request("track.getInfo", artist=artist, track=track, autocorrect=1)
    except Exception:
        data = {}
    _lastfm_track_cache[key] = data
    time.sleep(0.05)
    return data

# Genre buckets inspired by Rentfrow & Gosling macro‑dimensions and pragmatic categories
GENRE_BUCKETS: Dict[str, set] = {
    "reflective_complex": {
        "classical","baroque","romantic era","chamber","symphony","opera",
        "orchestral","choral","jazz","bebop","cool jazz","post-bop",
        "swing","latin jazz","fusion jazz","ambient","post rock","post-rock",
        "neo classical","neo-classical","instrumental","piano"
    },
    "intense_rebellious": {
        "metal","heavy metal","black metal","death metal","thrash","hardcore",
        "punk","emo","screamo","grindcore","industrial","alt metal"
    },
    "upbeat_conventional": {
        "country","adult contemporary","soft rock","oldies","singer songwriter",
        "singer-songwriter","easy listening","acoustic pop"
    },
    "energetic_rhythmic": {
        "edm","electronic","dance","house","techno","trance","drum and bass",
        "drum & bass","dnb","dubstep","garage","electro","bassline"
    },
    "hip_hop": {"hip hop","rap","trap","grime"},
    "rnb_soul": {"r&b","rnb","soul","neo-soul","neo soul"},
    "pop": {"pop","k pop","k-pop","j pop","j-pop","europop","indie pop","synthpop","dance pop"},
    "indie_alt": {"indie","indie rock","alternative","alt rock","alt-rock","shoegaze","lo fi","lo-fi"},
    "folk_acoustic": {"folk","indie folk","acoustic","americana"},
    "latin": {"latin","reggaeton","salsa","bachata","bossa nova","cumbia"},
}

def norm_token(s: str) -> str:
    """Normalise a genre or tag string for matching (lowercase, replace separators)."""
    return (s or "").lower().replace("-", " ").replace("/", " ").strip()

def map_tokens_to_buckets(tokens: List[str]) -> CounterType:
    """Map arbitrary genre/tag tokens to the defined genre buckets."""
    counts: CounterType = Counter()
    for raw in tokens:
        t = norm_token(raw)
        if not t:
            continue
        for bucket, vocab in GENRE_BUCKETS.items():
            if any(t == v or t.startswith(v) or v in t for v in vocab):
                counts[bucket] += 1
    return counts

# -----------------------------------------------------------------------------
# Spotify data collection functions
# -----------------------------------------------------------------------------

def fetch_user_profile(sp: spotipy.Spotify) -> Dict[str, Any]:
    """Fetch the current user's profile information."""
    try:
        return sp.current_user() or {}
    except Exception:
        return {}

def fetch_all_saved_tracks(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    """Retrieve all tracks saved in the user's library."""
    out: List[Dict[str, Any]] = []
    try:
        results = sp.current_user_saved_tracks(limit=50)
    except Exception:
        results = None
    while results:
        out.extend([it.get("track") for it in results.get("items", []) if it.get("track")])
        if results.get("next"):
            try:
                results = sp.next(results)
            except Exception:
                break
        else:
            break
    return out

def fetch_all_playlists_and_tracks(sp: spotipy.Spotify) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Retrieve all playlists owned or followed by the user and their tracks."""
    playlists: List[Dict[str, Any]] = []
    playlist_tracks: Dict[str, List[Dict[str, Any]]] = {}
    try:
        results = sp.current_user_playlists(limit=50)
    except Exception:
        results = None
    while results:
        items = results.get("items", []) if results else []
        playlists.extend(items)
        for pl in items:
            pid = pl.get("id")
            if not pid:
                continue
            tracks: List[Dict[str, Any]] = []
            try:
                res = sp.playlist_tracks(pid, limit=100)
            except Exception:
                res = None
            while res:
                tracks.extend([t.get("track") for t in res.get("items", []) if t.get("track")])
                if res.get("next"):
                    try:
                        res = sp.next(res)
                    except Exception:
                        break
                else:
                    break
            playlist_tracks[pid] = tracks
        if results and results.get("next"):
            try:
                results = sp.next(results)
            except Exception:
                break
        else:
            break
    return playlists, playlist_tracks

def fetch_all_followed_artists(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    """Retrieve all artists followed by the user."""
    out: List[Dict[str, Any]] = []
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

def fetch_top(sp: spotipy.Spotify) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Retrieve the user's top tracks and artists across different time ranges."""
    time_ranges = ["short_term", "medium_term", "long_term"]
    top_tracks: Dict[str, List[Dict[str, Any]]] = {}
    top_artists: Dict[str, List[Dict[str, Any]]] = {}
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

def fetch_recent(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    """Retrieve the user's recently played tracks (limited to last 50)."""
    try:
        return sp.current_user_recently_played(limit=50).get("items", [])
    except Exception:
        return []

# -----------------------------------------------------------------------------
# ID collection helpers
# -----------------------------------------------------------------------------

def collect_unique_track_ids(data: Dict[str, Any]) -> List[str]:
    """Collect a deduplicated list of track IDs from the collected data."""
    ids: List[str] = []
    # Top tracks
    for v in data.get("top_tracks", {}).values():
        ids.extend([t.get("id") for t in v if t and t.get("id") and t.get("type") == "track"])
    # Saved tracks
    for t in data.get("saved_tracks", []):
        if t and t.get("id") and t.get("type") == "track":
            ids.append(t["id"])
    # Playlist tracks
    for tracks in data.get("playlist_tracks", {}).values():
        ids.extend([t.get("id") for t in tracks if t and t.get("id") and t.get("type") == "track"])
    # Recently played
    for item in data.get("recent", []):
        tr = item.get("track")
        if tr and tr.get("id") and tr.get("type") == "track":
            ids.append(tr["id"])
    # Deduplicate and ensure non‑empty
    return list({tid for tid in ids if tid})

def collect_unique_artist_ids(data: Dict[str, Any]) -> List[str]:
    """Collect a deduplicated list of artist IDs from the collected data."""
    a_ids: List[str] = []
    # Top artists
    for v in data.get("top_artists", {}).values():
        a_ids.extend([a.get("id") for a in v if a and a.get("id")])
    # Artists from playlists and saved tracks
    for tracks in data.get("playlist_tracks", {}).values():
        for t in tracks:
            for a in t.get("artists", []):
                if a.get("id"):
                    a_ids.append(a["id"])
    for t in data.get("saved_tracks", []):
        for a in t.get("artists", []):
            if a.get("id"):
                a_ids.append(a["id"])
    # Artists from recent plays
    for item in data.get("recent", []):
        tr = item.get("track")
        if tr:
            for a in tr.get("artists", []):
                if a.get("id"):
                    a_ids.append(a["id"])
    return list({aid for aid in a_ids if aid})

# -----------------------------------------------------------------------------
# Enrichment functions
# -----------------------------------------------------------------------------

def fetch_track_details_and_features(
    sp: spotipy.Spotify,
    track_ids: List[str],
    include_audio_features: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch metadata for a list of track IDs, optionally including audio features.

    Due to Spotify removing public access to the `/v1/audio-features` endpoint
    for new applications (November 27 2024), audio feature retrieval is
    disabled by default.  If `include_audio_features` is True, the function
    attempts to request audio features in batches and falls back to
    per‑track requests when batch calls fail.  Any tracks that still
    produce errors during feature retrieval are silently skipped for
    audio data, leaving their feature fields unset.
    """
    details: Dict[str, Dict[str, Any]] = {}
    total = len(track_ids)
    ok_feature_tracks = 0
    prog = st.empty()

    for i, chunk in enumerate(batch(track_ids, 50), start=1):
        prog.text(
            f"Fetching track metadata{' and features' if include_audio_features else ''}: "
            f"batch {i}/{max(1, math.ceil(total/50))}"
        )
        try:
            tlist = sp.tracks(chunk).get("tracks", [])
        except Exception:
            tlist = []
        # Filter to true tracks only
        track_map = {t["id"]: t for t in tlist if t and t.get("type") == "track" and t.get("id")}
        valid_ids = list(track_map.keys())
        # Placeholder for features
        feats: List[Any] = [None] * len(valid_ids)
        if include_audio_features and valid_ids:
            try:
                tmp = sp.audio_features(valid_ids) or []
                if len(tmp) != len(valid_ids):
                    tmp += [None] * (len(valid_ids) - len(tmp))
                feats = tmp
                ok_feature_tracks += sum(1 for f in feats if f)
            except Exception:
                # Batch request failed (e.g. 403); fallback to per‑track
                fallback: List[Any] = []
                for tid in valid_ids:
                    try:
                        f_list = sp.audio_features([tid])
                        if f_list and f_list[0]:
                            fallback.append(f_list[0])
                            ok_feature_tracks += 1
                        else:
                            fallback.append(None)
                    except Exception:
                        fallback.append(None)
                feats = fallback
        # Compose metadata
        for tid, f_info in zip(valid_ids, feats):
            t_info = track_map.get(tid)
            if not t_info:
                continue
            artists = [a.get("name") for a in t_info.get("artists", []) if a.get("name")]
            meta: Dict[str, Any] = {
                "name": t_info.get("name"),
                "artists": artists,
                "album": t_info.get("album", {}).get("name"),
                "release_date": t_info.get("album", {}).get("release_date"),
                "popularity": t_info.get("popularity", 0),
            }
            if include_audio_features and f_info:
                for k in [
                    "danceability",
                    "energy",
                    "valence",
                    "acousticness",
                    "instrumentalness",
                    "speechiness",
                    "liveness",
                    "loudness",
                    "tempo",
                    "key",
                    "mode",
                    "time_signature",
                ]:
                    meta[k] = f_info.get(k)
            # Last.fm enrichment
            try:
                if artists and meta.get("name"):
                    lf_info = get_lastfm_track(artists[0], meta["name"])
                    track_block = lf_info.get("track", {})
                    playcount = track_block.get("playcount")
                    if playcount is not None:
                        meta["lfm_playcount"] = int(playcount)
                    tags = track_block.get("toptags", {}).get("tag", [])
                    if isinstance(tags, list):
                        meta["lfm_tags"] = [t.get("name") for t in tags[:10] if t.get("name")]
            except Exception:
                pass
            details[tid] = meta
    prog.empty()
    if include_audio_features:
        st.info(f"Audio features coverage: {ok_feature_tracks} / {total} tracks had features returned.")
    return details

def fetch_artist_details(
    sp: spotipy.Spotify,
    artist_ids: List[str],
    include_lastfm: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Fetch metadata for a list of artist IDs, optionally enriched with Last.fm info."""
    details: Dict[str, Dict[str, Any]] = {}
    total = len(artist_ids)
    prog = st.empty()
    for i, chunk in enumerate(batch(artist_ids, 50), start=1):
        prog.text(f"Fetching artist metadata: batch {i}/{max(1, math.ceil(total/50))}")
        try:
            arts = sp.artists(chunk).get("artists", [])
        except Exception:
            arts = []
        for a in arts:
            aid = a.get("id")
            if not aid:
                continue
            meta: Dict[str, Any] = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0),
            }
            if include_lastfm:
                try:
                    la = get_lastfm_artist(a.get("name") or "")
                    ablock = la.get("artist", {})
                    stats = ablock.get("stats", {})
                    pc = stats.get("playcount")
                    if pc is not None:
                        meta["lfm_playcount"] = int(pc)
                    tags = ablock.get("tags", {}).get("tag", [])
                    if isinstance(tags, list):
                        meta["lfm_tags"] = [t.get("name") for t in tags[:10] if t.get("name")]
                except Exception:
                    pass
            details[aid] = meta
    prog.empty()
    return details

# -----------------------------------------------------------------------------
# Personality inference
# -----------------------------------------------------------------------------

def safe_mean(vals: List[float]) -> float:
    """Compute the mean of a list of numeric values, ignoring non‑numerics."""
    nums = [v for v in vals if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else 0.0

def compute_personality(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute heuristic Big Five personality scores from the enriched data.

    The method aggregates genre and tag information from artists and tracks,
    computes genre bucket shares and diversity, and uses track popularity to
    assess mainstreamness.  If audio features are present for some tracks
    (which requires extended API access), they are blended into the scoring
    functions; otherwise, the heuristics rely solely on genres and tags.
    """
    tracks = list(data.get("track_details", {}).values())
    artists = list(data.get("artist_details", {}).values())
    # Collect tokens (Spotify genres + Last.fm tags)
    tokens: List[str] = []
    for a in artists:
        tokens.extend(a.get("genres", []) or [])
        tokens.extend(a.get("lfm_tags", []) or [])
    for t in tracks:
        tokens.extend(t.get("lfm_tags", []) or [])
    bucket_counts = map_tokens_to_buckets(tokens)
    total_hits = sum(bucket_counts.values()) or 1
    bucket_share = {k: v / total_hits for k, v in bucket_counts.items()}
    genre_diversity = len({norm_token(x) for x in tokens if x}) / (len(tokens) or 1)
    # Mainstreamness: average track popularity scaled to 0..1
    pops = [t.get("popularity") for t in tracks if isinstance(t.get("popularity"), (int, float))]
    mainstreamness = (sum(pops) / len(pops) / 100.0) if pops else 0.0
    # Determine whether audio features are available
    feature_fields = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "loudness",
        "tempo",
    ]
    coverage = {f: sum(1 for t in tracks if isinstance(t.get(f), (int, float))) for f in feature_fields}
    have_features = sum(coverage.values()) > 0
    avg: Dict[str, float] = {}
    loud_norm = tempo_norm = 0.0
    if have_features:
        for f in feature_fields:
            avg[f] = safe_mean([t.get(f) for t in tracks if isinstance(t.get(f), (int, float))])
        loud_norm = max(0.0, min(1.0, (avg.get("loudness", 0.0) + 60.0) / 60.0))
        tempo_norm = max(0.0, min(1.0, avg.get("tempo", 0.0) / 200.0))
    # Compute scores
    # Extraversion: upbeat & contemporary; emphasise energetic/rhythmic and pop; blend in danceability/valence if present
    extraversion = (
        ((avg.get("danceability", 0.0) + avg.get("valence", 0.0)) * 0.35 if have_features else 0) +
        (bucket_share.get("energetic_rhythmic", 0.0) + bucket_share.get("pop", 0.0)) * 0.65
    )
    # Conscientiousness: structured/mainstream, lower intensity & tempo, fewer intense/rebellious
    conscientiousness = (
        ((1 - avg.get("energy", 0.0)) * 0.20 if have_features else 0) +
        ((1 - tempo_norm) * 0.15 if have_features else 0) +
        (1 - bucket_share.get("intense_rebellious", 0.0)) * 0.40 +
        mainstreamness * 0.25
    )
    # Agreeableness: mellow, warm and acoustic; emphasise folk/acoustic and upbeat/conventional buckets
    agreeableness = (
        ((avg.get("acousticness", 0.0)) * 0.25 if have_features else 0) +
        ((1 - avg.get("energy", 0.0)) * 0.10 if have_features else 0) +
        (1 - (avg.get("speechiness", 0.0) if have_features else 0)) * 0.05 +
        (bucket_share.get("folk_acoustic", 0.0) + bucket_share.get("upbeat_conventional", 0.0)) * 0.60
    )
    # Openness: diversity and reflective/complex; emphasise reflective_complex bucket and genre diversity; add acoustic/instrumental if present
    openness = (
        genre_diversity * 0.40 +
        ((avg.get("acousticness", 0.0) + avg.get("instrumentalness", 0.0)) * 0.20 if have_features else 0) +
        ((1 - loud_norm) * 0.10 if have_features else 0) +
        ((1 - avg.get("energy", 0.0)) * 0.05 if have_features else 0) +
        bucket_share.get("reflective_complex", 0.0) * 0.25
    )
    # Neuroticism: intense & emotional; emphasise intense_rebellious bucket and low valence/danceability; some energy
    neuroticism = (
        ((1 - avg.get("danceability", 0.0)) * 0.25 if have_features else 0.15) +
        ((1 - avg.get("valence", 0.0)) * 0.25 if have_features else 0.15) +
        ((avg.get("energy", 0.0)) * 0.10 if have_features else 0) +
        bucket_share.get("intense_rebellious", 0.0) * 0.50
    )
    scores = {
        "Openness": max(0.0, min(1.0, openness)),
        "Conscientiousness": max(0.0, min(1.0, conscientiousness)),
        "Extraversion": max(0.0, min(1.0, extraversion)),
        "Agreeableness": max(0.0, min(1.0, agreeableness)),
        "Neuroticism": max(0.0, min(1.0, neuroticism)),
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

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main() -> None:
    sp = ensure_spotify_client()
    # Sidebar options
    st.sidebar.header("Options")
    include_features = st.sidebar.checkbox(
        "Include Spotify audio features (requires extended access)", value=False
    )
    include_lastfm = st.sidebar.checkbox("Include Last.fm enrichment", value=True)
    # Explanatory caption about disabled audio features.  Triple quotes allow
    # the string to span multiple lines without syntax errors.  Note that we
    # avoid using single quotes inside the string to prevent premature
    # termination.
    st.sidebar.caption(
        """Audio features have been disabled by default because Spotify’s audio‑feature
        endpoint was deprecated for new apps in Nov 2024. Last.fm enrichment
        provides genre tags and playcounts."""
    )
    if st.button("🔎 Pull data and infer personality"):
        with st.spinner("Fetching your Spotify profile…"):
            user = fetch_user_profile(sp)
        st.success(f"Hello, {user.get('display_name') or 'Spotify user'}! 👋")
        # Pull datasets
        with st.spinner("Pulling Spotify datasets…"):
            top_tracks, top_artists = fetch_top(sp)
            saved_tracks = fetch_all_saved_tracks(sp)
            playlists, playlist_tracks = fetch_all_playlists_and_tracks(sp)
            followed_artists = fetch_all_followed_artists(sp)
            recent = fetch_recent(sp)
        data: Dict[str, Any] = {
            "top_tracks": top_tracks,
            "top_artists": top_artists,
            "saved_tracks": saved_tracks,
            "playlists": playlists,
            "playlist_tracks": playlist_tracks,
            "followed_artists": followed_artists,
            "recent": recent,
        }
        # Collect IDs
        t_ids = collect_unique_track_ids(data)
        a_ids = collect_unique_artist_ids(data)
        st.info(f"Unique tracks: {len(t_ids)} | Unique artists: {len(a_ids)}")
        # Enrich tracks
        with st.spinner("Enriching tracks…"):
            track_details = fetch_track_details_and_features(sp, t_ids, include_audio_features=include_features)
            if not include_lastfm:
                for v in track_details.values():
                    v.pop("lfm_playcount", None)
                    v.pop("lfm_tags", None)
        data["track_details"] = track_details
        # Enrich artists
        with st.spinner("Enriching artists…"):
            artist_details = fetch_artist_details(sp, a_ids, include_lastfm=include_lastfm)
        data["artist_details"] = artist_details
        # Display sample tracks and artists
        st.subheader("🎼 Enriched Tracks (first 12)")
        for t in list(track_details.values())[:12]:
            tags = ", ".join(t.get("lfm_tags", []) or []) or "—"
            st.write(
                f"- {t.get('name')} — {', '.join(t.get('artists', []) or [])} "
                f"(Pop: {t.get('popularity')}) | Tags: {tags}"
            )
        st.subheader("🎤 Enriched Artists (first 12)")
        for a in list(artist_details.values())[:12]:
            genres = ", ".join(a.get("genres", []) or []) or "—"
            tags = ", ".join(a.get("lfm_tags", []) or []) or "—"
            st.write(
                f"- {a.get('name')} | Genres: {genres} | Pop: {a.get('popularity')} "
                f"| Followers: {a.get('followers')} | Last.fm plays: {a.get('lfm_playcount', '—')} | Tags: {tags}"
            )
        # Compute personality
        with st.spinner("Inferring personality…"):
            scores, debug = compute_personality(data)
        st.subheader("🧠 Personality (scores 0–1)")
        dominant_trait = max(scores.items(), key=lambda kv: kv[1])
        st.write(f"**Dominant trait:** {dominant_trait[0]} ({dominant_trait[1]:.2f})")
        for k, v in scores.items():
            st.write(f"- {k}: {v:.2f}")
        with st.expander("🔬 Debug information"):
            st.json(debug)
        with st.expander("📦 Summary counts"):
            st.json({
                "saved_tracks": len(saved_tracks),
                "playlists": len(playlists),
                "recent_items": len(recent),
                "followed_artists": len(followed_artists),
                "unique_track_ids": len(t_ids),
                "unique_artist_ids": len(a_ids),
            })

if __name__ == "__main__":
    main()