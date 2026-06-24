from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "init.sql"


def load_schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")
