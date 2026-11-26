"""Normalize immutable Silver news Parquet batches for Spark streaming.

Silver writes one Parquet file per batch. Some files contain nanosecond
Arrow timestamps, which Spark's Parquet reader cannot always decode. This
small Python sidecar reads each new Silver file with pandas/pyarrow, casts
its timestamps to microseconds, and atomically publishes a normalized file
into the Gold v4 streaming inbox.

The Silver folders are read-only. No Silver files are modified or moved.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import pandas as pd

from config import settings


class NewsSilverNormalizer:
    def __init__(self, poll_seconds: int | None = None):
        self.source_dir = Path(settings.SILVER_NEWS_BY_TICKER_PATH)
        self.output_dir = Path(settings.GOLD_V4_NEWS_INPUT_PATH)
        self.poll_seconds = int(
            poll_seconds or settings.GOLD_V4_NORMALIZER_INTERVAL_SECONDS
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _first_column(frame: pd.DataFrame, names: tuple[str, ...], required=False):
        lookup = {str(c).lower(): c for c in frame.columns}
        for name in names:
            if name.lower() in lookup:
                return lookup[name.lower()]
        if required:
            raise KeyError(
                f"Missing required Silver column; tried {names}; "
                f"available={list(frame.columns)}"
            )
        return None

    @staticmethod
    def _symbols(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw = list(value)
        elif hasattr(value, "tolist") and not isinstance(value, str):
            raw = value.tolist()
        elif isinstance(value, str):
            cleaned = value.strip().strip("[]")
            raw = [part.strip(" '\"\t") for part in cleaned.split(",")] if cleaned else []
        else:
            raw = [value]

        symbols = []
        for item in raw:
            if item is None:
                continue
            symbol = str(item).strip().upper()
            for suffix in (".NS", ".BO", ".NSE", ".BSE"):
                if symbol.endswith(suffix):
                    symbol = symbol[: -len(suffix)]
                    break
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return symbols

    def _normalize_frame(self, source_file: Path) -> pd.DataFrame:
        frame = pd.read_parquet(source_file)
        if frame.empty:
            return pd.DataFrame()

        article_col = self._first_column(frame, ("article_id", "id"), required=True)
        ticker_col = self._first_column(frame, ("ticker", "symbol", "stock_symbol"))
        matched_col = self._first_column(frame, ("matched_tickers", "matched_ticker"))
        headline_col = self._first_column(frame, ("headline", "cleaned_headline", "title"))
        source_col = self._first_column(frame, ("source", "publisher"))
        sentiment_col = self._first_column(frame, ("sentiment", "sentiment_label", "label"))
        score_col = self._first_column(frame, ("sentiment_score", "score", "compound"))
        positive_col = self._first_column(frame, ("positive_score",))
        negative_col = self._first_column(frame, ("negative_score",))
        neutral_col = self._first_column(frame, ("neutral_score",))
        published_col = self._first_column(frame, ("published_at", "publication_timestamp"))

        configured_event_col = str(settings.NEWS_TIME_COLUMN or "ingested_at")
        event_col = self._first_column(
            frame,
            (configured_event_col, "ingested_at", "published_at"),
            required=True,
        )

        rows = pd.DataFrame(index=frame.index)
        rows["article_id"] = frame[article_col].astype("string").str.strip()

        if ticker_col:
            rows["symbol"] = frame[ticker_col].apply(lambda value: self._symbols(value)[:1])
        elif matched_col:
            rows["symbol"] = frame[matched_col].apply(self._symbols)
        else:
            raise KeyError("Silver news needs ticker or matched_tickers")

        rows = rows.explode("symbol", ignore_index=False)
        rows["symbol"] = rows["symbol"].astype("string").str.strip().str.upper()
        # Reindex by the original row index after explode so a single article
        # with several symbols keeps its original article-level attributes.
        original_index = rows.index
        if headline_col:
            rows["headline"] = frame.loc[original_index, headline_col].astype("string").to_numpy()
        else:
            rows["headline"] = None
        if source_col:
            rows["source"] = frame.loc[original_index, source_col].astype("string").to_numpy()
        else:
            rows["source"] = None
        if sentiment_col:
            rows["sentiment"] = frame.loc[original_index, sentiment_col].astype("string").to_numpy()
        else:
            rows["sentiment"] = None

        def numeric(column):
            if column is None:
                return pd.Series([None] * len(rows), index=rows.index, dtype="object")
            return pd.to_numeric(frame.loc[original_index, column], errors="coerce").reset_index(drop=True)

        rows["sentiment_score"] = numeric(score_col).to_numpy()
        rows["positive_score"] = numeric(positive_col).to_numpy()
        rows["negative_score"] = numeric(negative_col).to_numpy()
        rows["neutral_score"] = numeric(neutral_col).to_numpy()

        if published_col:
            rows["published_at"] = pd.to_datetime(
                frame.loc[original_index, published_col], errors="coerce", utc=True
            ).dt.tz_convert(None).astype("datetime64[us]").to_numpy()
        else:
            rows["published_at"] = pd.NaT

        rows["event_time"] = pd.to_datetime(
            frame.loc[original_index, event_col], errors="coerce", utc=True
        ).dt.tz_convert(None).astype("datetime64[us]").to_numpy()

        # A malformed ingestion timestamp can fall back to publication time.
        missing_event = pd.isna(rows["event_time"])
        if missing_event.any():
            rows.loc[missing_event, "event_time"] = rows.loc[missing_event, "published_at"]

        rows["processed_at"] = pd.Timestamp.now(tz="UTC").tz_convert(None).to_datetime64()
        rows = rows.dropna(subset=["article_id", "symbol", "event_time"])
        rows = rows[rows["article_id"].astype(str).str.len() > 0]
        rows = rows[rows["symbol"].astype(str).str.len() > 0]
        rows = rows.drop_duplicates(subset=["article_id", "symbol"], keep="last")

        return rows[[
            "article_id", "symbol", "headline", "source", "published_at",
            "event_time", "sentiment", "sentiment_score", "positive_score",
            "negative_score", "neutral_score", "processed_at",
        ]].reset_index(drop=True)

    @staticmethod
    def _write_microsecond_parquet(frame: pd.DataFrame, target: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        data = frame.copy()
        for column in ("article_id", "symbol", "headline", "source", "sentiment"):
            data[column] = data[column].astype("string")
        for column in ("sentiment_score", "positive_score", "negative_score", "neutral_score"):
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
        for column in ("published_at", "event_time", "processed_at"):
            data[column] = pd.to_datetime(data[column], errors="coerce").astype("datetime64[us]")

        table = pa.Table.from_pandas(data, preserve_index=False)
        for column in ("published_at", "event_time", "processed_at"):
            index = table.schema.get_field_index(column)
            arrow_values = pa.array(data[column], type=pa.timestamp("us"), from_pandas=True)
            table = table.set_column(index, column, arrow_values)

        temporary = target.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, target)

    def scan_once(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            print(f"[GOLD V4][NEWS NORMALIZER] Source not found: {self.source_dir}")
            return 0

        candidates = sorted(self.source_dir.rglob("*.parquet"))
        written = 0

        for source_file in candidates:
            try:
                relative = source_file.relative_to(self.source_dir).as_posix()
                stat = source_file.stat()
                identity = f"{relative}|{stat.st_size}|{stat.st_mtime_ns}"
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                target = self.output_dir / f"news_{digest}.parquet"
                if target.exists():
                    continue

                normalized = self._normalize_frame(source_file)
                if normalized.empty:
                    # An empty normalized file is not emitted. The source is
                    # retried on the next scan in case it was still being written.
                    continue

                self._write_microsecond_parquet(normalized, target)
                written += len(normalized)
                print(
                    f"[GOLD V4][NEWS NORMALIZER] {source_file.name}: "
                    f"{len(normalized)} article-symbol rows -> {target.name}"
                )
            except Exception as exc:
                print(f"[GOLD V4][NEWS NORMALIZER] Skipped {source_file}: {exc}")

        return written

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while not self._stop.wait(self.poll_seconds):
                self.scan_once()

        self._stop.clear()
        self._thread = threading.Thread(
            target=loop, name="fintel-gold-v4-news-normalizer", daemon=True
        )
        self._thread.start()
        print(
            f"[GOLD V4][NEWS NORMALIZER] Watching {self.source_dir} "
            f"every {self.poll_seconds}s"
        )

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(2, self.poll_seconds + 1))
