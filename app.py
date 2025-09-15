"""
Streamlit application for analysing a user's Spotify listening habits and
inferring their dominant Big Five personality trait.  The application
authenticates with Spotify using OAuth, retrieves as much data as
possible about the user's listening history (top tracks and artists,
saved tracks, playlists and their tracks, recently played tracks and
followed artists), enriches that data with acoustic features from
Spotify as well as additional metadata from the Last.fm API, and then
computes simple heuristic scores for each of the Big Five personality
dimensions.  The personality heuristics implemented here are grounded
in published research on the relationship between personality and
musical preferences.

Important research foundations:

* Extraversion is positively associated with contemporary, upbeat and
  danceable music.  In a cross‑cultural study covering more than 50
  countries, Greenberg et al. (2022) found positive correlations
  between extraversion and contemporary music【427880637911948†L63-L67】.  At a
  finer granularity, Melchiorre and Schedl (2020) showed that
  extraversion correlates positively with average danceability and
  valence of the tracks a person listens to【934760148766067†L361-L369】.

* Conscientiousness tends to correlate negatively with intense,
  aggressive musical styles【427880637911948†L63-L66】.  People scoring high on
  conscientiousness also show less diverse listening behaviour and
  favour uncomplicated, unpretentious music.  In the audio feature
  domain this corresponds to lower energy and less variability in
  loudness【934760148766067†L355-L369】.

* Agreeableness is associated with mellow, romantic and acoustic
  music【427880637911948†L63-L67】.  Melchiorre and Schedl (2020) observed
  positive correlations between agreeableness and acousticness along
  with negative correlations with energy and speechiness【934760148766067†L370-L390】.

* Openness (sometimes called open‑mindedness) is linked to a
  preference for complex, sophisticated and diverse music, including
  classical, jazz and world genres【427880637911948†L63-L67】.  On the
  audio‑feature level, openness correlates positively with
  acousticness and instrumentalness and negatively with loudness,
  energy and tempo; open people also tend to have more diverse
  listening habits【934760148766067†L336-L347】.

* Neuroticism (negative emotionality) has been found to correlate
  negatively with danceability and positively with intense music
  styles【934760148766067†L392-L396】.  Neurotic individuals often consume
  more music overall and prefer tracks with lower valence, which
  reflect and sometimes help regulate negative moods.

The heuristic scoring functions in this application are simplistic and
serve as an example of how one might operationalise these findings in
code.  They should not be interpreted as clinical assessments.
"""

from __future__ import annotations

import os
import math
import json
from collections import defaultdict
from typing import Dict, List, Any, Iterator

import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

try:
    # If python‑dotenv is available, load environment variables from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Proceed silently if dotenv is not installed
    pass


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
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

st.set_page_config(
    page_title="Spotify + Last.fm Data Viewer", page_icon="🎧", layout="wide"
)
st.title("🎧 Spotify + Last.fm Data Viewer")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error(
        "Missing Spotify credentials. Set SPOTIPY_CLIENT_ID, "
        "SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI."
    )
    st.stop()
if not LASTFM_API_KEY:
    st.error("Missing Last.fm credentials. Set LASTFM_API_KEY.")
    st.stop()


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
def get_auth_manager() -> SpotifyOAuth:
    """Instantiate a SpotifyOAuth manager with our configured settings."""
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
    """Exchange an authorization code for an access token.

    Spotipy changed its API in version 2.22, so we handle both return types.
    """
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
    """Ensure there is a valid Spotify client for the current session.

    This function checks for a stored token in Streamlit's session state and
    refreshes it if necessary.  If no valid token is available it initiates
    the OAuth flow by prompting the user to log in via Spotify.
    """
    auth_manager = get_auth_manager()
    token_info = st.session_state.get("token_info")

    # If we already have a token and it hasn't expired, return a Spotify client
    if token_info and not auth_manager.is_token_expired(token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    # If token expired, attempt refresh
    if token_info and auth_manager.is_token_expired(token_info):
        try:
            st.session_state["token_info"] = auth_manager.refresh_access_token(
                token_info["refresh_token"]
            )
            return spotipy.Spotify(
                auth=st.session_state["token_info"]["access_token"]
            )
        except Exception:
            # Remove expired/invalid token and fall back to login flow
            st.session_state.pop("token_info", None)

    # Check for callback code in URL parameters
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

    # Otherwise prompt user to login
    login_url = auth_manager.get_authorize_url()
    st.info("You need to log in with Spotify to continue.")
    st.markdown(f"[Log in with Spotify]({login_url})")
    st.stop()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def lastfm_request(method: str, **params: Any) -> Dict[str, Any]:
    """Send a request to the Last.fm API and return the JSON response.

    Raises requests.HTTPError on failure.
    """
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "method": method,
        **params,
    }
    response = requests.get(base_url, params=query, timeout=10)
    response.raise_for_status()
    return response.json()


def batch(iterable: List[Any], n: int = 50) -> Iterator[List[Any]]:
    """Yield successive n‑sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------
def fetch_full_user_data(sp: spotipy.Spotify) -> Dict[str, Any]:
    """Fetch a comprehensive set of the user's Spotify data and enrich it.

    This function retrieves the user's top tracks and artists over three
    different time ranges, all saved tracks, all playlists and their tracks,
    recently played items and followed artists.  It then enriches this data
    with Spotify audio features for each unique track and with additional
    metadata from the Last.fm API for both tracks and artists.  The return
    value is a dictionary keyed by data category.
    """
    data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Top tracks and artists
    # ------------------------------------------------------------------
    time_ranges = ["short_term", "medium_term", "long_term"]
    data["top_tracks"] = {}
    data["top_artists"] = {}
    for tr in time_ranges:
        try:
            data["top_tracks"][tr] = sp.current_user_top_tracks(
                limit=50, time_range=tr
            ).get("items", [])
        except Exception:
            data["top_tracks"][tr] = []
        try:
            data["top_artists"][tr] = sp.current_user_top_artists(
                limit=50, time_range=tr
            ).get("items", [])
        except Exception:
            data["top_artists"][tr] = []

    # ------------------------------------------------------------------
    # Saved tracks (no artificial cap)
    # ------------------------------------------------------------------
    saved_tracks: List[Dict[str, Any]] = []
    try:
        results = sp.current_user_saved_tracks(limit=50)
    except Exception:
        results = None
    while results:
        # Each item has a "track" field holding the track object
        saved_tracks.extend(
            [item.get("track") for item in results.get("items", []) if item.get("track")]
        )
        if results.get("next"):
            try:
                results = sp.next(results)
            except Exception:
                break
        else:
            break
    data["saved_tracks"] = saved_tracks

    # ------------------------------------------------------------------
    # Playlists and their tracks
    # ------------------------------------------------------------------
    playlists: List[Dict[str, Any]] = []
    playlist_tracks: Dict[str, List[Dict[str, Any]]] = {}
    try:
        results = sp.current_user_playlists(limit=50)
    except Exception:
        results = None
    while results:
        items = results.get("items", []) if results else []
        playlists.extend(items)
        # For each playlist, fetch all its tracks
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
                tracks.extend(
                    [t.get("track") for t in res.get("items", []) if t.get("track")]
                )
                if res.get("next"):
                    try:
                        res = sp.next(res)
                    except Exception:
                        break
                else:
                    break
            playlist_tracks[pid] = tracks
        # Move to next page of playlists
        if results and results.get("next"):
            try:
                results = sp.next(results)
            except Exception:
                break
        else:
            break
    data["playlists"] = playlists
    data["playlist_tracks"] = playlist_tracks

    # ------------------------------------------------------------------
    # Followed artists
    # ------------------------------------------------------------------
    followed_artists: List[Dict[str, Any]] = []
    try:
        results = sp.current_user_followed_artists(limit=50)
    except Exception:
        results = None
    # Note: current_user_followed_artists returns a dict with 'artists'
    # key which itself has 'items' and 'next'.  The 'after' parameter in
    # subsequent requests is provided in the query string of the next URL.
    while results:
        artists_data = results.get("artists", {}) if results else {}
        followed_artists.extend(artists_data.get("items", []))
        next_url = artists_data.get("next")
        if next_url:
            # Spotipy doesn't expose a convenience method for this, but
            # sp._get follows pagination URLs internally.
            try:
                results = sp._get(next_url)  # type: ignore
            except Exception:
                break
        else:
            break
    data["followed_artists"] = followed_artists

    # ------------------------------------------------------------------
    # Recently played tracks (limited by Spotify to last 50)
    # ------------------------------------------------------------------
    try:
        data["recent"] = sp.current_user_recently_played(limit=50).get("items", [])
    except Exception:
        data["recent"] = []

    # ------------------------------------------------------------------
    # Compile unique track IDs from all collected sources
    # ------------------------------------------------------------------
    track_ids: List[str] = []
    # top tracks
    for v in data["top_tracks"].values():
        track_ids.extend([t.get("id") for t in v if t and t.get("id")])
    # saved tracks
    track_ids.extend([t.get("id") for t in saved_tracks if t and t.get("id")])
    # playlist tracks
    for tracks in playlist_tracks.values():
        track_ids.extend([t.get("id") for t in tracks if t and t.get("id")])
    # recently played tracks
    track_ids.extend(
        [item.get("track", {}).get("id") for item in data["recent"] if item.get("track")]
    )
    # Deduplicate
    track_ids = list({tid for tid in track_ids if tid})

    # ------------------------------------------------------------------
    # Fetch track details and audio features
    #
    # Spotify's audio features endpoint returns a 403 if any of the IDs in
    # the query are invalid (e.g. None). To avoid this we filter the list
    # and only call audio_features with valid string IDs.  We zip only over
    # the valid IDs to maintain alignment between IDs, track info and
    # features.
    # ------------------------------------------------------------------
    track_details: Dict[str, Dict[str, Any]] = {}
    for chunk in batch(track_ids, 50):
        # Filter out invalid IDs to prevent 403 errors
        valid_ids = [tid for tid in chunk if tid and isinstance(tid, str)]
        if not valid_ids:
            continue
        # Fetch basic track info for valid IDs
        try:
            tracks = sp.tracks(valid_ids).get("tracks", [])
        except Exception:
            tracks = []
        # Fetch audio features for valid IDs
        try:
            feats = sp.audio_features(valid_ids) or []
        except Exception:
            feats = [None] * len(valid_ids)
        for tid, t_info, f_info in zip(valid_ids, tracks, feats):
            if not t_info:
                continue
            artists_names = [a.get("name") for a in t_info.get("artists", [])]
            track_meta = {
                "name": t_info.get("name"),
                "artists": artists_names,
                "album": t_info.get("album", {}).get("name"),
                "release_date": t_info.get("album", {}).get("release_date"),
                "popularity": t_info.get("popularity", 0),
                "duration_ms": t_info.get("duration_ms"),
            }
            # Include audio features if available
            audio_fields = [
                "danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "duration_ms",
                "key",
                "mode",
                "time_signature",
            ]
            for field in audio_fields:
                track_meta[field] = f_info.get(field) if f_info else None
            # Enrich track with Last.fm data (playcount and tags)
            lastfm_info = {}
            if artists_names and track_meta.get("name"):
                try:
                    lf_resp = lastfm_request(
                        "track.getInfo",
                        track=track_meta["name"],
                        artist=artists_names[0],
                        autocorrect=1,
                    )
                    if "track" in lf_resp:
                        lf_track = lf_resp.get("track", {})
                        playcount = lf_track.get("playcount")
                        if playcount is not None:
                            lastfm_info["lfm_playcount"] = int(playcount)
                        tags = lf_track.get("toptags", {}).get("tag", [])
                        if isinstance(tags, list):
                            lastfm_info["lfm_tags"] = [t.get("name") for t in tags[:10]]
                except Exception:
                    pass
            track_meta.update(lastfm_info)
            track_details[tid] = track_meta
    data["track_details"] = track_details

    # ------------------------------------------------------------------
    # Compile unique artist IDs from all collected sources
    # ------------------------------------------------------------------
    artist_ids: List[str] = []
    for v in data["top_artists"].values():
        artist_ids.extend([a.get("id") for a in v if a and a.get("id")])
    for tracks in playlist_tracks.values():
        for t in tracks:
            for a in t.get("artists", []):
                if a.get("id"):
                    artist_ids.append(a["id"])
    for t in saved_tracks:
        for a in t.get("artists", []):
            if a.get("id"):
                artist_ids.append(a["id"])
    for item in data["recent"]:
        for a in item.get("track", {}).get("artists", []):
            if a.get("id"):
                artist_ids.append(a["id"])
    # Deduplicate
    artist_ids = list({aid for aid in artist_ids if aid})

    # ------------------------------------------------------------------
    # Fetch artist details (Spotify and Last.fm)
    # ------------------------------------------------------------------
    artist_details: Dict[str, Dict[str, Any]] = {}
    for chunk in batch(artist_ids, 50):
        try:
            res = sp.artists(chunk).get("artists", [])
        except Exception:
            res = []
        for a in res:
            aid = a.get("id")
            if not aid:
                continue
            details = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0),
            }
            # Last.fm enrichment: playcount and tags
            try:
                lfm = lastfm_request("artist.getInfo", artist=a.get("name"))
                if "artist" in lfm:
                    lf_artist = lfm.get("artist", {})
                    stats = lf_artist.get("stats", {})
                    playcount = stats.get("playcount")
                    if playcount is not None:
                        details["lfm_playcount"] = int(playcount)
                    tags_list = lf_artist.get("tags", {}).get("tag", [])
                    if isinstance(tags_list, list):
                        details["lfm_tags"] = [t.get("name") for t in tags_list[:10]]
            except Exception:
                pass
            artist_details[aid] = details
    data["artist_details"] = artist_details

    return data


# -----------------------------------------------------------------------------
# Personality inference
# -----------------------------------------------------------------------------
def _safe_mean(values: List[float]) -> float:
    """Return the mean of a list of numeric values, ignoring None entries."""
    vals = [v for v in values if isinstance(v, (int, float))]
    return float(sum(vals)) / len(vals) if vals else 0.0


def compute_personality_scores(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute heuristic Big Five personality scores from the enriched data.

    The heuristics are based on correlations reported in the literature.
    Each score is scaled between 0 and 1.  The highest score indicates the
    personality dimension most strongly suggested by the user's listening
    behaviour.
    """
    tracks = data.get("track_details", {}).values()
    if not tracks:
        return {t: 0.0 for t in ["extraversion", "conscientiousness", "agreeableness", "openness", "neuroticism"]}

    # Gather lists of audio feature values across all tracks
    danceabilities = [t.get("danceability") for t in tracks]
    energies = [t.get("energy") for t in tracks]
    valences = [t.get("valence") for t in tracks]
    acousticness = [t.get("acousticness") for t in tracks]
    instrumentalness = [t.get("instrumentalness") for t in tracks]
    speechiness = [t.get("speechiness") for t in tracks]
    loudness = [t.get("loudness") for t in tracks]
    tempos = [t.get("tempo") for t in tracks]

    # Compute averages
    avg_danceability = _safe_mean(danceabilities)
    avg_energy = _safe_mean(energies)
    avg_valence = _safe_mean(valences)
    avg_acousticness = _safe_mean(acousticness)
    avg_instrumentalness = _safe_mean(instrumentalness)
    avg_speechiness = _safe_mean(speechiness)
    avg_loudness = _safe_mean(loudness)
    avg_tempo = _safe_mean(tempos)

    # Normalise loudness (which is typically between -60 and 0 dB) to [0,1]
    if avg_loudness is not None:
        loudness_norm = max(0.0, min(1.0, (avg_loudness + 60.0) / 60.0))
    else:
        loudness_norm = 0.0
    # Normalise tempo: assume 50–200 BPM typical range
    if avg_tempo is not None:
        tempo_norm = max(0.0, min(1.0, avg_tempo / 200.0))
    else:
        tempo_norm = 0.0

    # Compute genre diversity: number of unique genres / total genre assignments
    genre_list: List[str] = []
    for artist_id in data.get("artist_details", {}):
        artist_genres = data["artist_details"][artist_id].get("genres", [])
        genre_list.extend(artist_genres)
    unique_genres = set(g for g in genre_list if g)
    diversity_score = 0.0
    if genre_list:
        diversity_score = len(unique_genres) / len(genre_list)

    # Heuristic scores
    # Extraversion: danceability and valence are positively correlated【934760148766067†L361-L369】
    extraversion = (avg_danceability + avg_valence) / 2.0

    # Conscientiousness: negative correlation with intense music (energy) and high tempo【427880637911948†L63-L66】
    conscientiousness = 1.0 - (avg_energy + tempo_norm) / 2.0

    # Agreeableness: positive correlation with acousticness and low energy, low speechiness【934760148766067†L370-L390】
    agreeableness = (
        avg_acousticness + (1.0 - avg_energy) + (1.0 - avg_speechiness)
    ) / 3.0

    # Openness: positive correlation with acousticness, instrumentalness and diversity; negative with loudness and energy【934760148766067†L336-L347】
    openness = (
        avg_acousticness
        + avg_instrumentalness
        + diversity_score
        + (1.0 - loudness_norm)
        + (1.0 - avg_energy)
    ) / 5.0

    # Neuroticism: negative correlation with danceability and valence, positive with energy【934760148766067†L392-L396】
    neuroticism = ((1.0 - avg_danceability) + (1.0 - avg_valence) + avg_energy) / 3.0

    # Ensure scores are within [0,1]
    scores = {
        "extraversion": max(0.0, min(1.0, extraversion)),
        "conscientiousness": max(0.0, min(1.0, conscientiousness)),
        "agreeableness": max(0.0, min(1.0, agreeableness)),
        "openness": max(0.0, min(1.0, openness)),
        "neuroticism": max(0.0, min(1.0, neuroticism)),
    }
    return scores


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def render_data_and_personality(data: Dict[str, Any]) -> None:
    """Render the enriched user data and personality inference to the UI."""
    # Display a subset of enriched track details
    st.subheader("🎼 Enriched Tracks (sample)")
    for i, t in enumerate(list(data.get("track_details", {}).values())[:10]):
        st.write(
            f"- {t['name']} — {', '.join(t['artists'])} "
            f"(Album: {t['album']}, Release: {t['release_date']}, "
            f"Energy: {t.get('energy')}, Valence: {t.get('valence')}, "
            f"Danceability: {t.get('danceability')}, Tempo: {t.get('tempo')}, "
            f"Acousticness: {t.get('acousticness')}, Instrumentalness: {t.get('instrumentalness')})"
        )

    # Display a subset of enriched artist details
    st.subheader("🎤 Enriched Artists (sample)")
    for a in list(data.get("artist_details", {}).values())[:10]:
        genres = ", ".join(a.get("genres", [])) or "N/A"
        tags = ", ".join(a.get("lfm_tags", [])) if a.get("lfm_tags") else "N/A"
        st.write(
            f"- {a['name']} | Genres: {genres} | Popularity: {a['popularity']} | "
            f"Followers: {a['followers']} | Last.fm playcount: {a.get('lfm_playcount', 'N/A')} | "
            f"Tags: {tags}"
        )

    # Compute and display personality scores
    scores = compute_personality_scores(data)
    st.subheader("🧠 Personality Inference (heuristic)")
    if scores:
        # Determine the dominant trait
        dominant_trait = max(scores.items(), key=lambda x: x[1])[0]
        st.markdown(
            f"**Predicted dominant trait:** `{dominant_trait.title()}`\n"
            "The scores below range from 0 (low) to 1 (high) and represent how strongly your listening patterns align with each trait."
        )
        # Display scores
        for trait, score in scores.items():
            st.write(f"- {trait.title()}: {score:.2f}")
    else:
        st.info("Not enough track data available to compute personality scores.")

    # Show raw data as JSON inside an expander
    with st.expander("📦 Raw JSON data"):
        st.json(data)


def main() -> None:
    """Main entry point of the Streamlit app."""
    sp = ensure_spotify_client()
    if st.button("🔎 Pull my Spotify + Last.fm data"):
        with st.spinner("Fetching your Spotify + Last.fm data..."):
            data = fetch_full_user_data(sp)
        render_data_and_personality(data)


if __name__ == "__main__":
    main()