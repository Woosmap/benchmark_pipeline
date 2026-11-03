from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

DATA.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
REPO_DIR = Path(__file__).resolve()

GOOGLE_PLACES_FILE = DATA / "google-data" / "google_places_752452_1698_951_13.jl"