"""SQLite 检测历史持久化模块。

采用 threading.Lock 保证多线程写安全，init_db 内存缓存避免重复建表。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_initialized: set[str] = set()  # 已初始化的数据库路径缓存，避免每次操作都 CREATE TABLE IF NOT EXISTS


def _db_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / "detection_history.db"


def init_db(data_dir: Path) -> None:
    """首次调用时建表，后续调用直接跳过（基于内存缓存）。"""
    key = str(data_dir.resolve())
    if key in _initialized:
        return
    path = _db_path(data_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                filename TEXT,
                conf REAL,
                iou REAL,
                image_width INTEGER,
                image_height INTEGER,
                thumb_jpeg_base64 TEXT,
                detections_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
    _initialized.add(key)


def add_record(
    data_dir: Path,
    *,
    filename: str | None,
    conf: float,
    iou: float,
    image_width: int,
    image_height: int,
    thumb_jpeg_base64: str | None,
    detections: list[dict[str, Any]],
) -> int:
    """写入一条检测记录，返回自增 ID。"""
    path = _db_path(data_dir)
    payload = json.dumps(detections, ensure_ascii=False)
    with _lock, sqlite3.connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO records (created_at, filename, conf, iou, image_width, image_height, thumb_jpeg_base64, detections_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                filename or "",
                conf,
                iou,
                image_width,
                image_height,
                thumb_jpeg_base64 or "",
                payload,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_records(
    data_dir: Path,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """分页查询历史列表，不含 detections_json（体积大，列表页不需要）。

    返回 {"records": [...], "total": int}
    """
    path = _db_path(data_dir)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        rows = conn.execute(
            "SELECT id, created_at, filename, conf, iou, image_width, image_height, "
            "thumb_jpeg_base64, detections_json "
            "FROM records ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        dets = json.loads(r["detections_json"])
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "filename": r["filename"],
                "conf": r["conf"],
                "iou": r["iou"],
                "image_width": r["image_width"],
                "image_height": r["image_height"],
                "thumb_jpeg_base64": r["thumb_jpeg_base64"] or None,
                "num_detections": len(dets),
            }
        )
    return {"records": out, "total": total}


def get_record(data_dir: Path, record_id: int) -> dict[str, Any] | None:
    """查询单条记录详情（含完整 detections，用于详情弹窗）。"""
    path = _db_path(data_dir)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "conf": row["conf"],
        "iou": row["iou"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "thumb_jpeg_base64": row["thumb_jpeg_base64"] or None,
        "detections": json.loads(row["detections_json"]),
    }


def delete_record(data_dir: Path, record_id: int) -> bool:
    """删除记录，返回是否成功。"""
    path = _db_path(data_dir)
    with _lock, sqlite3.connect(path) as conn:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        return cur.rowcount > 0
