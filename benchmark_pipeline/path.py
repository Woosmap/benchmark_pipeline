from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

DATA.mkdir(parents=True, exist_ok=True)