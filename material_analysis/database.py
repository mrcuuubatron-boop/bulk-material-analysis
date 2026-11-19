"""SQLite persistence for pipeline runs and measurements."""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from .models import Measurement


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    input_dir TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    valid_count INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    avg_image_ms REAL NOT NULL,
    materials TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    area_px REAL NOT NULL,
    width_px REAL NOT NULL,
    height_px REAL NOT NULL,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    mean_gray REAL NOT NULL,
    std_gray REAL NOT NULL,
    valid INTEGER NOT NULL,
    predicted_material TEXT NOT NULL DEFAULT '',
    confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_measurements_run_id ON measurements(run_id);
"""


class StatisticsDatabase:
    def __init__(self, path: str | Path = "output/statistics.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def record_run(
        self,
        mode: str,
        input_dir: str | Path,
        output_dir: str | Path,
        measurements: Iterable[Measurement],
        duration_seconds: float,
        predictions: Iterable[dict] = (),
    ) -> int:
        rows = list(measurements)
        prediction_by_file = {row["filename"]: row for row in predictions}
        materials = sorted({
            row.get("predicted_material", "")
            for row in prediction_by_file.values()
            if row.get("predicted_material")
        })
        avg_image_ms = duration_seconds * 1000 / len(rows) if rows else 0.0
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    started_at, mode, input_dir, output_dir, image_count,
                    valid_count, duration_seconds, avg_image_ms, materials
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    mode,
                    str(input_dir),
                    str(output_dir),
                    len(rows),
                    sum(row.valid for row in rows),
                    duration_seconds,
                    avg_image_ms,
                    json.dumps(materials, ensure_ascii=False),
                ),
            )
            run_id = int(cursor.lastrowid)
            for row in rows:
                prediction = prediction_by_file.get(row.filename, {})
                connection.execute(
                    """
                    INSERT INTO measurements (
                        run_id, filename, area_px, width_px, height_px, cx, cy,
                        mean_gray, std_gray, valid, predicted_material, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        row.filename,
                        row.area_px,
                        row.width_px,
                        row.height_px,
                        row.cx,
                        row.cy,
                        row.mean_gray,
                        row.std_gray,
                        int(row.valid),
                        prediction.get("predicted_material", ""),
                        prediction.get("confidence"),
                    ),
                )
        return run_id

    def summary(self) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS runs, COALESCE(SUM(image_count), 0) AS images,
                       COALESCE(SUM(valid_count), 0) AS valid_images,
                       COALESCE(AVG(duration_seconds), 0) AS avg_duration,
                       COALESCE(AVG(avg_image_ms), 0) AS avg_image_ms
                FROM runs
                """
            ).fetchone()
            return dict(row)

    def runs(self, limit: int = 100) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, started_at, mode, input_dir, image_count, valid_count,
                       duration_seconds, avg_image_ms, materials
                FROM runs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def material_statistics(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT predicted_material AS material,
                       COUNT(*) AS images,
                       SUM(valid) AS valid_images,
                       AVG(area_px) AS avg_area_px,
                       AVG(confidence) AS avg_confidence
                FROM measurements
                WHERE predicted_material <> ''
                GROUP BY predicted_material
                ORDER BY images DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]
