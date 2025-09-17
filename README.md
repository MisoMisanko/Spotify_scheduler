# Spotify Personality Prediction

This repository contains the code and analysis for my MSc dissertation
project. The aim is to explore whether personality traits can be
predicted from music listening patterns and to prototype an application
that could extend this concept into festival activity recommendations.

## Project Overview

The project has two main components:

1.  **Machine Learning Analysis**
    -   Trains models to predict Big Five personality traits from either
        music genre preferences or audio features.\
    -   Uses validated public datasets (Kaggle Young People Survey and
        Figshare PER dataset).\
    -   Compares genre-based vs. audio-feature-based approaches.
2.  **Streamlit Application**
    -   Full app coded in Streamlit.\
    -   Spotify login to collect personal listening history.\
    -   Enriches Spotify data with Last.fm and MusicBrainz tags.\
    -   Analyzes listening consistency and diversity across time
        periods.\
    -   Generates personality predictions and visualizes results in a
        radar chart.\
    -   Deployed with `secrets.toml` in Streamlit Cloud for managing
        credentials (Spotify and Last.fm API keys).

## Repository Structure

    SPOTIFY_SCHEDULER/
    │  app.py                      # Streamlit app
    │  production_predictor.py      # Model loading and prediction interface
    │  requirements.txt             # Python dependencies
    │
    ├─ data/
    │   ├─ kaggle_young_people/     # Kaggle Young People Survey dataset
    │   └─ figshare_PER/            # PER dataset
    │
    ├─ models/
    │   ├─ production_personality_models.pkl     # Trained model package
    │   └─ production_model_performance.csv      # Model performance summary
    │
    └─ src/
        ├─ final_analysis.py         # Complete analysis and comparison
        └─ train_production_model.py # Training script for production models

## Installation

1.  Clone this repository:

    ``` bash
    git clone https://github.com/your-username/SPOTIFY_SCHEDULER.git
    cd SPOTIFY_SCHEDULER
    ```

2.  Create and activate a virtual environment:

    ``` bash
    python -m venv venv
    source venv/bin/activate    # on macOS/Linux
    venv\Scripts\activate       # on Windows
    ```

3.  Install the required packages:

    ``` bash
    pip install -r requirements.txt
    ```

## Usage

### Running Locally

The application is designed to run on **Streamlit Cloud**, where API
secrets are stored securely in the platform's `secrets.toml`.

If you want to run the app locally, you must create a `.env` file in the
project root and add your API keys:

    SPOTIPY_CLIENT_ID=your_spotify_client_id
    SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
    SPOTIPY_REDIRECT_URI=http://localhost:8501/callback
    LASTFM_API_KEY=your_lastfm_api_key

Then launch the Streamlit app:

``` bash
streamlit run app.py
```

### Training Models

To retrain the models on the Kaggle dataset:

``` bash
python src/train_production_model.py
```

### Running Analysis

To reproduce the dissertation analysis:

``` bash
python src/final_analysis.py
```

## Requirements

All dependencies are listed in `requirements.txt`.

## Acknowledgements

-   Kaggle Young People Survey dataset\
-   PER dataset (Figshare)\
-   Spotify API, Last.fm API, and MusicBrainz API

## Disclaimer

This project was developed as part of an MSc dissertation. Large
Language Models (LLMs), including ChatGPT and Claude, were used to help
develop, structure, and debug the code, as well as to draft
documentation and supporting materials.
