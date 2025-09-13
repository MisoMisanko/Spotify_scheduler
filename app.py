import os
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Attempt to load environment variables from a .env file if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python‑dotenv isn't installed the environment variables must be
    # provided another way; silently continue.
    pass


# ---------------------------------------------------------------------------
# Configuration
#
# The app requires four scopes to fetch listening history and playlist data.
# ``user-top-read`` permits reading the user's top artists and tracks across
# multiple time ranges, ``playlist-read-private`` allows reading private
# playlists, ``user-library-read`` fetches saved tracks, and ``user-follow-read``
# allows reading collaborative playlist metadata.  These scopes are defined in
# the SCOPES constant below.  If you change the scopes you must also update
# the permissions requested in Spotify Developer Dashboard.
# ---------------------------------------------------------------------------
SCOPES = "user-top-read playlist-read-private user-library-read user-follow-read"

# Read credentials from environment.  These variables *must* be defined in
# ``.streamlit/secrets.toml`` on Streamlit Cloud or exported in your local
# environment before running the app.
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error(
        "Spotify credentials are not configured. Please set SPOTIPY_CLIENT_ID, "
        "SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI in your environment "
        "variables or Streamlit secrets."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Helper functions
#
# A handful of simple helpers are defined up front.  They compute entropy,
# normalise values to 0–1, and perform light data wrangling.  The personality
# scoring later depends on these helpers to stay readable.
# ---------------------------------------------------------------------------
def shannon_entropy(counter: Counter) -> float:
    """Compute the Shannon entropy of a Counter object.  Returns 0 if empty."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12, 2)
    return entropy


def normalize(x: float, lo: float, hi: float) -> float:
    """Normalise a value x to the range [0, 1] given lower and upper bounds."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def year_from_release_date(rd: str) -> int | None:
    """
    Extract the year from a release date string.  Spotify dates may appear
    as 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'.  If parsing fails return None.
    """
    if not rd:
        return None
    try:
        return int(rd[:4])
    except Exception:
        return None


def batch(iterable, n: int = 50):
    """
    Yield successive n-sized chunks from an iterable.  Spotify's batch API
    endpoints accept up to 50 IDs at once.
    """
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def safe_artist_ids(tracks):
    """
    Extract the first artist ID from each track object if present.  Tracks
    without artists or IDs are skipped.  This helper is used when building
    the list of unique artists to fetch with sp.artists().
    """
    ids: list[str] = []
    for t in tracks:
        artists = t.get("artists", [])
        if artists:
            aid = artists[0].get("id")
            if aid:
                ids.append(aid)
    return ids


# Define some genre groupings to use when inferring agreeableness.  These are
# deliberately coarse buckets; the heuristics use relative proportions
# (e.g. mellow vs aggressive vs dancepop) rather than absolute counts.
MELLOW_GENRES = {"acoustic", "folk", "singer-songwriter", "ambient", "chill", "lo-fi", "soft rock"}
AGGRESSIVE_GENRES = {"metal", "hardcore", "punk", "industrial"}
DANCE_POP_GENRES = {"dance pop", "edm", "house", "techno", "trance"}


def genre_bucket_counts(genres: list[str]) -> dict[str, int]:
    """
    Categorise a list of genre strings into mellow, aggressive and dancepop
    buckets.  Genres that don't match these categories still contribute to
    the total count.  The matching is case-insensitive and checks for
    substring containment within the genre label.
    """
    buckets: dict[str, int] = defaultdict(int)
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


def classify_mbti(o: float, e: float, a: float, c: float) -> str:
    """
    Map normalised Big‑Five trait scores into a simple MBTI‑style code.
    This mapping is heuristic and intended for fun rather than clinical use.
    - Extraversion (E) vs Introversion (I)
    - Intuition (N) vs Sensing (S) maps from Openness
    - Feeling (F) vs Thinking (T) maps from Agreeableness vs Conscientiousness
    - Judging (J) vs Perceiving (P) maps from Conscientiousness vs Openness
    """
    ei = "E" if e >= 0.5 else "I"
    ns = "N" if o >= 0.5 else "S"
    ft = "F" if a >= 0.5 else "T"
    jp = "J" if c >= 0.5 else "P"
    return ei + ns + ft + jp


# ---------------------------------------------------------------------------
# Token management
#
# The standard Spotipy OAuth flow is not well suited to Streamlit Cloud
# because it expects to run a local HTTP server to capture the redirect.  The
# ``open_browser=False`` flag prevents that, while ``show_dialog=True``
# ensures that the user always sees a login prompt (which is useful for
# incognito sessions).  After logging in, Spotify appends a ``code``
# parameter to the callback URL; Streamlit makes these query parameters
# accessible via ``st.experimental_get_query_params()``.  The function
# ``get_token()`` below exchanges the code for an access token and stores
# both the token and the refresh token in ``st.session_state``.  When a
# valid token already exists the function returns it immediately; if it has
# expired Spotipy refreshes it.
# ---------------------------------------------------------------------------
def get_token() -> str:
    """
    Retrieve a valid access token.  If the user has not logged in yet, show a
    login link and stop execution.  Tokens and refresh tokens are stored in
    ``st.session_state`` for the duration of the Streamlit session.
    """
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=".cache",  # not used on Streamlit Cloud but kept for local dev
        open_browser=False,
        show_dialog=True,
    )

    # If we've already stored a token, check its validity.
    token_info = st.session_state.get("token_info")
    if token_info:
        if not auth_manager.is_token_expired(token_info):
            return token_info["access_token"]
        # Refresh the token if expired
        try:
            token_info = auth_manager.refresh_access_token(token_info["refresh_token"])
            st.session_state["token_info"] = token_info
            return token_info["access_token"]
        except Exception as exc:
            # If refreshing fails, clear the session state so we fall through to login
            st.session_state.pop("token_info", None)

    # See if the ``code`` parameter is present in the URL (user has just authenticated)
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        try:
            token_info = auth_manager.get_access_token(code)
        except Exception as exc:
            st.error(f"Failed to exchange code for access token: {exc}")
            st.stop()
        st.session_state["token_info"] = token_info
        return token_info["access_token"]

    # No token and no code: prompt the user to log in.
    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue. Please click the link below.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()


# ---------------------------------------------------------------------------
# Data collection and analysis
#
# The functions below fetch user data from Spotify and compute a variety of
# summary statistics.  We limit the number of API calls to keep the app
# responsive by only fetching up to 20 top tracks and artists per time range
# and up to 100 saved tracks.  Playlists are limited to the first 20.  The
# resulting signals are normalised and fed into our heuristic trait model.
# ---------------------------------------------------------------------------

def fetch_user_data(sp: spotipy.Spotify) -> dict:
    """
    Fetch a number of datasets from Spotify: top tracks and artists over
    multiple time ranges, a sample of saved tracks, user's playlists and
    detailed artist information.  The returned dictionary contains these
    fields:

      ``top_tracks``: dict mapping time range to list of track objects
      ``top_artists``: dict mapping time range to list of artist objects
      ``saved_tracks``: list of up to 100 saved track objects
      ``playlists``: list of up to 20 playlist objects
      ``artist_details``: dict mapping artist ID to details (popularity, followers, genres)

    This function can be expensive because of multiple API calls, so avoid
    calling it repeatedly.  Consider caching its result with Streamlit's
    ``st.cache_data`` decorator if running locally.
    """
    data: dict[str, dict] = {}
    time_ranges = ["short_term", "medium_term", "long_term"]
    # Fetch top tracks and artists (limit to 20 to improve responsiveness)
    data["top_tracks"] = {tr: sp.current_user_top_tracks(limit=20, time_range=tr).get("items", []) for tr in time_ranges}
    data["top_artists"] = {tr: sp.current_user_top_artists(limit=20, time_range=tr).get("items", []) for tr in time_ranges}

    # Fetch saved tracks (sample up to 100)
    saved: list[dict] = []
    try:
        results = sp.current_user_saved_tracks(limit=50)
        saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
        if results.get("next"):
            # Fetch one additional page to get up to 100 saved tracks
            results = sp.next(results)
            saved.extend([it["track"] for it in results.get("items", []) if "track" in it])
    except Exception:
        # On any API error return what we have
        pass
    data["saved_tracks"] = saved[:100]

    # Fetch playlists (limit to 20 for speed)
    try:
        pls = sp.current_user_playlists(limit=20).get("items", [])
    except Exception:
        pls = []
    data["playlists"] = pls

    # Build set of unique artist IDs across top tracks, saved tracks and top artists
    artist_ids: set[str] = set()
    for tr in time_ranges:
        artist_ids.update([a.get("id") for a in data["top_artists"][tr]])
        artist_ids.update(safe_artist_ids(data["top_tracks"][tr]))
    artist_ids.update(safe_artist_ids(data["saved_tracks"]))
    # Remove None values
    artist_ids.discard(None)  # type: ignore

    # Fetch details for each artist in batches
    artist_details: dict[str, dict] = {}
    for b in batch(list(artist_ids), 50):
        try:
            res = sp.artists(b).get("artists", [])
            for a in res:
                artist_details[a["id"]] = {
                    "popularity": a.get("popularity", 0),
                    "followers": a.get("followers", {}).get("total", 0),
                    "genres": a.get("genres", []),
                }
        except Exception:
            continue
    data["artist_details"] = artist_details
    return data


def compute_signals(data: dict) -> dict:
    """
    Given the raw data dictionary from ``fetch_user_data``, compute a variety of
    summary statistics.  The returned dictionary includes both raw signals
    (such as genre entropy, average popularity, long track share) and
    intermediate indices (e.g. stability vs novelty).  These signals feed
    directly into the trait scoring functions.
    """
    signals: dict[str, float] = {}

    # Assemble all unique tracks across top tracks and saved tracks
    unique_tracks: list[dict] = []
    seen_ids: set[str] = set()
    for tracks in data["top_tracks"].values():
        for t in tracks:
            tid = t.get("id")
            if tid and tid not in seen_ids:
                unique_tracks.append(t)
                seen_ids.add(tid)
    for t in data.get("saved_tracks", []):
        tid = t.get("id")
        if tid and tid not in seen_ids:
            unique_tracks.append(t)
            seen_ids.add(tid)

    # Extract artist genres, popularities and follower counts
    artist_genres: list[str] = []
    artist_pops: list[int] = []
    artist_follows: list[int] = []
    for det in data["artist_details"].values():
        artist_pops.append(det.get("popularity", 0))
        artist_follows.append(det.get("followers", 0))
        artist_genres.extend(det.get("genres", []))

    # Genre entropy and diversity
    genre_counter = Counter([g.lower() for g in artist_genres])
    signals["genre_entropy"] = shannon_entropy(genre_counter)
    unique_genres = len(genre_counter)
    total_genres = sum(genre_counter.values())
    signals["genre_diversity"] = unique_genres / total_genres if total_genres else 0.0

    # Popularity & mainstream vs niche
    signals["avg_artist_popularity"] = float(np.mean(artist_pops)) if artist_pops else 0.0
    signals["niche_artist_share"] = (
        sum(1 for p in artist_pops if p < 40) / len(artist_pops)
        if artist_pops
        else 0.0
    )
    signals["long_tail_share"] = (
        sum(1 for f in artist_follows if f < 100_000) / len(artist_follows)
        if artist_follows
        else 0.0
    )

    # Track duration profile
    durations_min = [t.get("duration_ms", 0) / 60000.0 for t in unique_tracks if t.get("duration_ms")]
    signals["avg_duration_min"] = float(np.mean(durations_min)) if durations_min else 0.0
    signals["long_track_share"] = (
        sum(1 for d in durations_min if d >= 6.0) / len(durations_min)
        if durations_min
        else 0.0
    )

    # Recency of listening: share of tracks released within the last 2 years
    years = []
    for t in unique_tracks:
        year = year_from_release_date(t.get("album", {}).get("release_date"))
        if year:
            years.append(year)
    current_year = datetime.utcnow().year
    signals["recency_share"] = (
        sum(1 for y in years if (current_year - y) <= 2) / len(years)
        if years
        else 0.0
    )

    # Stability vs novelty: intersection of short vs long term top artists
    short_ids = {a.get("id") for a in data["top_artists"].get("short_term", [])}
    long_ids = {a.get("id") for a in data["top_artists"].get("long_term", [])}
    intersection = len(short_ids & long_ids)
    union = max(1, len(short_ids | long_ids))
    signals["stability_index"] = intersection / union
    signals["novelty_index"] = 1.0 - signals["stability_index"]

    # Social/collaborative cues: proportion of collaborative playlists
    playlists = data.get("playlists", [])
    collab_count = sum(1 for p in playlists if p.get("collaborative"))
    signals["collab_playlist_ratio"] = (
        collab_count / len(playlists)
        if playlists
        else 0.0
    )

    # Genre buckets
    buckets = genre_bucket_counts(list(genre_counter.keys()))
    total_bucket = max(1, buckets.get("total", 0))
    signals["mellow_ratio"] = buckets.get("mellow", 0) / total_bucket
    signals["aggressive_ratio"] = buckets.get("aggressive", 0) / total_bucket
    signals["dancepop_ratio"] = buckets.get("dancepop", 0) / total_bucket

    return signals


def compute_traits(signals: dict) -> dict[str, float]:
    """
    Convert the raw signal dictionary into four normalised trait scores.  These
    weights were chosen heuristically; you may adjust them to emphasise
    different aspects of listening behaviour.  All scores are bounded in
    [0, 1].
    """
    openness = (
        0.45 * normalize(signals["genre_entropy"], 0.0, 8.0)
        + 0.25 * normalize(signals["genre_diversity"], 0.0, 0.7)
        + 0.15 * normalize(signals["long_track_share"], 0.0, 0.5)
        + 0.15 * normalize(signals["novelty_index"], 0.0, 1.0)
    )
    extraversion = (
        0.50 * normalize(signals["avg_artist_popularity"], 20.0, 80.0)
        + 0.25 * normalize(signals["collab_playlist_ratio"], 0.0, 0.5)
        + 0.25 * normalize(signals["recency_share"], 0.0, 0.8)
    )
    agreeableness = (
        0.65 * normalize(signals["mellow_ratio"] - signals["aggressive_ratio"], -0.5, 0.5)
        + 0.35 * normalize(signals["dancepop_ratio"], 0.0, 0.5)
    )
    conscientiousness = (
        0.7 * normalize(signals["stability_index"], 0.0, 1.0)
        + 0.3 * (1.0 - normalize(signals["long_track_share"], 0.0, 0.5))
    )
    return {
        "Openness": openness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Conscientiousness": conscientiousness,
    }


# ---------------------------------------------------------------------------
# Streamlit page layout
#
# The remainder of the script uses Streamlit to manage the UI flow.  The user
# must log in before clicking the analysis button; this is enforced by
# ``get_token()``.  Once the token is available the app fetches the data,
# computes signals and traits and displays charts and recommendations.  Raw
# signal values are tucked into an expander for transparency.  A radar chart
# offers a compact visual summary of the four traits.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Festival Personality App", page_icon="🎧", layout="centered")
st.title("🎧 Festival Personality App")

# Try to retrieve a valid token.  If this returns, the user is logged in.
token: str = get_token()
sp_client = spotipy.Spotify(auth=token)

# Once logged in show the analysis button
if st.button("🔎 Analyze my Spotify"):
    with st.spinner("Fetching your Spotify data and analysing it..."):
        data = fetch_user_data(sp_client)
        signals = compute_signals(data)
        traits = compute_traits(signals)
        mbti_code = classify_mbti(
            traits["Openness"], traits["Extraversion"], traits["Agreeableness"], traits["Conscientiousness"]
        )

    # Display trait scores as metrics
    st.subheader("🎭 Personality scores")
    cols = st.columns(4)
    for i, (trait_name, score) in enumerate(traits.items()):
        with cols[i]:
            st.metric(trait_name, f"{score:.2f}")

    # Create a radar chart of the four traits
    labels = list(traits.keys())
    values = list(traits.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="teal", alpha=0.25)
    ax.plot(angles, values, color="teal", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_title("Personality radar", pad=20)
    st.pyplot(fig)

    # MBTI code display
    st.markdown(
        f"""
        <div style='text-align:center; padding:20px; border-radius:10px;
        background-color:#f0f9f9; border:2px solid #00b3b3;'>
        <h2 style='color:#007777;'>Your MBTI Type</h2>
        <h1 style='color:#00b3b3;'>{mbti_code}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Provide festival recommendations based on trait scores
    st.subheader("🎶 Festival Recommendations")
    recommendations: list[str] = []
    # Openness
    if traits["Openness"] > 0.6:
        recommendations.append("🎨 Explore discovery zones & art installations")
    else:
        recommendations.append("🎤 Stick with headliners & familiar stages")
    # Extraversion
    if traits["Extraversion"] > 0.6:
        recommendations.append("💃 Join big dance tents & group activities")
    else:
        recommendations.append("🧘 Relax in acoustic/chill‑out areas")
    # Conscientiousness
    if traits["Conscientiousness"] > 0.6:
        recommendations.append("📅 Plan workshops & follow a structured schedule")
    else:
        recommendations.append("✨ Go with the flow & explore pop‑up events")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    # Raw signal values in an expander for transparency
    with st.expander("🔍 See raw analysis signals"):
        display_signals = {k: round(v, 3) for k, v in signals.items()}
        st.json(display_signals)
