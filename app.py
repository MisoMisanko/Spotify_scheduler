# app.py
import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

# Load environment variables if .env is present
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
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

st.set_page_config(page_title="Spotify + Last.fm Data Viewer", page_icon="🎧", layout="centered")
st.title("🎧 Spotify + Last.fm Data Viewer")

if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
    st.error("Missing Spotify credentials. Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI.")
    st.stop()
if not LASTFM_API_KEY:
    st.warning("⚠️ No Last.fm API key found. Set LASTFM_API_KEY to enrich with tags & playcounts.")

# -----------------------------
# Helpers
# -----------------------------
def batch(iterable, n=50):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def lastfm_request(method: str, **params):
    """Generic Last.fm API request helper."""
    base_url = "http://ws.audioscrobbler.com/2.0/"
    query = {
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "method": method,
        **params
    }
    r = requests.get(base_url, params=query, timeout=10)
    r.raise_for_status()
    return r.json()

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

# -----------------------------
# Data Fetch + Enrichment
# -----------------------------
def fetch_full_user_data(sp):
    data = {}
    time_ranges = ["short_term", "medium_term", "long_term"]

    # --- Top tracks & artists ---
    data["top_tracks"] = {tr: sp.current_user_top_tracks(limit=30, time_range=tr)["items"] for tr in time_ranges}
    data["top_artists"] = {tr: sp.current_user_top_artists(limit=30, time_range=tr)["items"] for tr in time_ranges}

    # --- Saved tracks (up to 200) ---
    saved = []
    results = sp.current_user_saved_tracks(limit=50)
    saved.extend([it["track"] for it in results.get("items", [])])
    while results.get("next") and len(saved) < 200:
        results = sp.next(results)
        saved.extend([it["track"] for it in results.get("items", [])])
    data["saved_tracks"] = saved

    # --- Playlists ---
    data["playlists"] = sp.current_user_playlists(limit=20)["items"]

    # --- Collect IDs ---
    track_ids, artist_ids = set(), set()
    for tr in time_ranges:
        for t in data["top_tracks"][tr]:
            if t.get("id"): track_ids.add(t["id"])
            for a in t.get("artists", []):
                if a.get("id"): artist_ids.add(a["id"])
        for a in data["top_artists"][tr]:
            if a.get("id"): artist_ids.add(a["id"])
    for t in saved:
        if t.get("id"): track_ids.add(t["id"])
        for a in t.get("artists", []):
            if a.get("id"): artist_ids.add(a["id"])

    # --- Enrich tracks ---
    
    track_details = {}
    for b in batch(list(track_ids), 50):
        try:
            feats = sp.audio_features(b)
        except Exception:
            feats = [None] * len(b)  # if request fails, pad with None

        try:
            tracks = sp.tracks(b)["tracks"]
        except Exception:
            tracks = [None] * len(b)

        for t, f in zip(tracks, feats):
            if not t:
                continue
            track_details[t["id"]] = {
                "name": t.get("name"),
                "artists": [a["name"] for a in t.get("artists", [])],
                "album": t.get("album", {}).get("name"),
                "release_date": t.get("album", {}).get("release_date"),
                "popularity": t.get("popularity", 0),
                "duration_ms": t.get("duration_ms"),
                "energy": f.get("energy") if f else None,
                "valence": f.get("valence") if f else None,
                "danceability": f.get("danceability") if f else None,
                "tempo": f.get("tempo") if f else None,
            }
    data["track_details"] = track_details


    # --- Enrich artists ---
    artist_details = {}
    for b in batch(list(artist_ids), 50):
        res = sp.artists(b)["artists"]
        for a in res:
            artist_info = {
                "name": a.get("name"),
                "genres": a.get("genres", []),
                "popularity": a.get("popularity", 0),
                "followers": a.get("followers", {}).get("total", 0)
            }
            # Last.fm enrichment
            if LASTFM_API_KEY:
                try:
                    res = lastfm_request("artist.getInfo", artist=a.get("name"))
                    if "artist" in res:
                        lfm = res["artist"]
                        artist_info["lfm_playcount"] = int(lfm.get("stats", {}).get("playcount", 0))
                        tags = [t["name"] for t in lfm.get("tags", {}).get("tag", [])]
                        artist_info["lfm_tags"] = tags[:5]
                except Exception:
                    pass
            artist_details[a["id"]] = artist_info
    data["artist_details"] = artist_details

    return data

# -----------------------------
# Main UI
# -----------------------------
sp = ensure_spotify_client()

if st.button("🔎 Pull my Spotify + Last.fm data"):
    with st.spinner("Fetching your Spotify + Last.fm data..."):
        data = fetch_full_user_data(sp)

    # --- Show Top Tracks ---
    st.subheader("🎵 Top Tracks by Range")
    for tr, tracks in data["top_tracks"].items():
        st.write(f"**{tr}**")
        for t in tracks[:5]:
            st.write("-", t["name"], "—", ", ".join(a["name"] for a in t["artists"]))

    # --- Show Top Artists ---
    st.subheader("🎤 Top Artists by Range")
    for tr, arts in data["top_artists"].items():
        st.write(f"**{tr}**")
        for a in arts[:5]:
            st.write("-", a["name"], "Genres:", ", ".join(a["genres"]))

    # --- Show Saved Tracks ---
    st.subheader("💾 Saved Tracks (sample)")
    for t in data["saved_tracks"][:10]:
        st.write("-", t["name"], "—", ", ".join(a["name"] for a in t["artists"]))

    # --- Show Playlists ---
    st.subheader("📑 Playlists")
    for p in data["playlists"]:
        st.write("-", p["name"], f"({p['tracks']['total']} tracks)")

    # --- Enriched Tracks ---
    st.subheader("🎼 Enriched Tracks (sample)")
    for tid, t in list(data["track_details"].items())[:10]:
        st.write(f"- {t['name']} — {', '.join(t['artists'])} "
                 f"(Energy: {t['energy']}, Valence: {t['valence']}, Danceability: {t['danceability']}, Tempo: {t['tempo']})")

    # --- Enriched Artists ---
    st.subheader("👩‍🎤 Enriched Artists (sample)")
    for aid, a in list(data["artist_details"].items())[:10]:
        tags = ", ".join(a.get("lfm_tags", [])) if "lfm_tags" in a else "N/A"
        st.write(f"- {a['name']} | Genres: {', '.join(a['genres'])} "
                 f"| Followers: {a['followers']} | Tags: {tags} "
                 f"| Last.fm playcount: {a.get('lfm_playcount', 'N/A')}")

    # --- Raw data ---
    with st.expander("📦 Raw JSON"):
        st.json(data)
