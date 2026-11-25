"""FinTel Gold v4.

Three independent Structured Streaming queries:
  1. Stock windows -> gold_stock_metrics (all completed stock windows)
  2. News articles -> gold_news_sentiment (one row per article-symbol)
  3. Windowed stock/news stream-stream INNER JOIN -> gold_sentiment_price

The integration is event-time based. Both sides use the configured 5-minute
window, 1-minute slide and 10-minute watermark. A joined row means the news
article(s) and stock ticks belong to the same symbol and exact event-time
window. Unmatched stock windows remain available in gold_stock_metrics and
can be left-joined by the dashboard; they are not fabricated as news matches.

Gold v4 writes directly to the existing PostgreSQL schema. It does not write
Gold Delta/Parquet output and does not reuse v1-v3 checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from config import settings
from gold import gold_storage


NEWS_SCHEMA = StructType([
    StructField("article_id", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("headline", StringType(), True),
    StructField("source", StringType(), True),
    StructField("published_at", TimestampType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("sentiment", StringType(), True),
    StructField("sentiment_score", DoubleType(), True),
    StructField("positive_score", DoubleType(), True),
    StructField("negative_score", DoubleType(), True),
    StructField("neutral_score", DoubleType(), True),
    StructField("processed_at", TimestampType(), True),
])


def _require_columns(df: DataFrame, required: set[str], source_name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}")


def _build_stock_events(ticks_raw: DataFrame) -> DataFrame:
    _require_columns(
        ticks_raw,
        {"symbol", "current_price", "change_percent", "publication_timestamp"},
        "Stock Silver",
    )

    event_time = F.coalesce(
        F.col("publication_timestamp"),
        F.col("market_timestamp") if "market_timestamp" in ticks_raw.columns else F.lit(None).cast("timestamp"),
    )

    return (
        ticks_raw
        .select(
            F.upper(F.trim(F.col("symbol"))).alias("symbol"),
            F.col("current_price").cast("double").alias("price"),
            F.col("change_percent").cast("double").alias("day_change_pct"),
            event_time.alias("event_time"),
        )
        .where(F.col("symbol").isNotNull() & (F.length("symbol") > 0))
        .where(F.col("price").isNotNull() & (F.col("price") > 0))
        .where(F.col("event_time").isNotNull())
    )


def _build_news_events(news_raw: DataFrame) -> DataFrame:
    _require_columns(
        news_raw,
        {"article_id", "symbol", "event_time", "published_at", "sentiment_score"},
        "Normalized News Silver",
    )

    return (
        news_raw
        .select(
            F.trim(F.col("article_id")).alias("article_id"),
            F.upper(F.trim(F.col("symbol"))).alias("symbol"),
            F.col("headline").cast("string").alias("headline"),
            F.col("source").cast("string").alias("source"),
            F.col("published_at").cast("timestamp").alias("published_at"),
            F.col("event_time").cast("timestamp").alias("event_time"),
            F.col("sentiment").cast("string").alias("sentiment"),
            F.col("sentiment_score").cast("double").alias("sentiment_score"),
            F.col("positive_score").cast("double").alias("positive_score"),
            F.col("negative_score").cast("double").alias("negative_score"),
            F.col("neutral_score").cast("double").alias("neutral_score"),
            F.col("processed_at").cast("timestamp").alias("processed_at"),
        )
        .where(F.col("article_id").isNotNull() & (F.length("article_id") > 0))
        .where(F.col("symbol").isNotNull() & (F.length("symbol") > 0))
        .where(F.col("event_time").isNotNull())
    )


def _build_stock_windows(stock_events: DataFrame) -> DataFrame:
    events = stock_events.withWatermark("event_time", settings.WATERMARK)
    return (
        events
        .groupBy(
            F.col("symbol"),
            F.window(
                "event_time", settings.WINDOW_DURATION, settings.SLIDE_DURATION
            ).alias("w"),
        )
        .agg(
            F.min_by("price", "event_time").alias("price_first"),
            F.max_by("price", "event_time").alias("price_last"),
            F.max("price").alias("price_high"),
            F.min("price").alias("price_low"),
            F.count(F.lit(1)).cast("long").alias("tick_count"),
            F.max_by("day_change_pct", "event_time").alias("day_change_pct"),
        )
    )


def _build_news_windows(news_events: DataFrame) -> DataFrame:
    events = news_events.withWatermark("event_time", settings.WATERMARK)
    return (
        events
        .groupBy(
            F.col("symbol"),
            F.window(
                "event_time", settings.WINDOW_DURATION, settings.SLIDE_DURATION
            ).alias("w"),
        )
        .agg(
            F.count(F.lit(1)).cast("int").alias("article_count"),
            F.avg("sentiment_score").alias("avg_sentiment"),
            F.min("sentiment_score").alias("min_sentiment"),
            F.max_by("headline", "event_time").alias("latest_headline"),
            F.max_by("sentiment", "event_time").alias("latest_sentiment_label"),
            F.max("event_time").alias("latest_news_time"),
        )
    )


def _stock_metrics_output(stock_windows: DataFrame) -> DataFrame:
    change_pct = F.when(
        F.col("price_first") != 0,
        ((F.col("price_last") - F.col("price_first")) / F.col("price_first")) * 100.0,
    )
    range_pct = F.when(
        F.col("price_first") != 0,
        ((F.col("price_high") - F.col("price_low")) / F.col("price_first")) * 100.0,
    )

    return stock_windows.select(
        F.col("symbol"),
        F.col("w.start").alias("window_start"),
        F.col("w.end").alias("window_end"),
        F.col("price_first"),
        F.col("price_last"),
        F.col("price_high"),
        F.col("price_low"),
        F.round(change_pct, 4).alias("price_change_pct"),
        F.round(range_pct, 4).alias("price_range_pct"),
        F.col("day_change_pct"),
        F.col("tick_count").cast("long"),
        F.current_timestamp().alias("computed_at"),
    )


def _build_integrated_stream(stock_windows: DataFrame, news_windows: DataFrame) -> DataFrame:
    stock = stock_windows.alias("s")
    news = news_windows.alias("n")

    # Exact equality on symbol + Spark event-time window is the temporal
    # constraint for the stream-stream join. Inner join keeps semantics
    # explicit: only same-window news is counted as an impact candidate.
    joined = stock.join(
        news,
        (F.col("s.symbol") == F.col("n.symbol"))
        & (F.col("s.w") == F.col("n.w")),
        "inner",
    )

    price_change = F.when(
        F.col("s.price_first") != 0,
        ((F.col("s.price_last") - F.col("s.price_first")) / F.col("s.price_first")) * 100.0,
    )
    price_range = F.when(
        F.col("s.price_first") != 0,
        ((F.col("s.price_high") - F.col("s.price_low")) / F.col("s.price_first")) * 100.0,
    )
    negative_sentiment = F.col("n.avg_sentiment") <= F.lit(settings.ALERT_SENTIMENT_THRESHOLD)
    price_drop = price_change <= F.lit(settings.ALERT_PRICE_DROP_PCT)
    alert_flag = F.coalesce(negative_sentiment | price_drop, F.lit(False))

    alert_reason = (
        F.when(
            negative_sentiment & price_drop,
            F.concat(
                F.lit("Negative sentiment and price drop; sentiment="),
                F.round(F.col("n.avg_sentiment"), 3).cast("string"),
                F.lit("; price change="),
                F.round(price_change, 3).cast("string"),
                F.lit("%"),
            ),
        )
        .when(
            negative_sentiment,
            F.concat(
                F.lit("Negative news sentiment: "),
                F.round(F.col("n.avg_sentiment"), 3).cast("string"),
            ),
        )
        .when(
            price_drop,
            F.concat(
                F.lit("Price drop: "),
                F.round(price_change, 3).cast("string"),
                F.lit("%"),
            ),
        )
    )

    return joined.select(
        F.col("s.symbol").alias("symbol"),
        F.col("s.w.start").alias("window_start"),
        F.col("s.w.end").alias("window_end"),
        F.col("n.article_count").cast("int").alias("article_count"),
        F.col("n.avg_sentiment").alias("avg_sentiment"),
        F.col("n.min_sentiment").alias("min_sentiment"),
        F.col("n.latest_headline").alias("latest_headline"),
        F.col("n.latest_sentiment_label").alias("latest_sentiment_label"),
        F.col("n.latest_news_time").alias("latest_news_time"),
        F.col("s.price_first").alias("price_first"),
        F.col("s.price_last").alias("price_last"),
        F.col("s.price_high").alias("price_high"),
        F.col("s.price_low").alias("price_low"),
        F.round(price_change, 4).alias("price_change_pct"),
        F.round(price_range, 4).alias("price_range_pct"),
        F.col("s.day_change_pct").alias("day_change_pct"),
        F.col("s.tick_count").cast("long").alias("tick_count"),
        alert_flag.alias("alert_flag"),
        alert_reason.alias("alert_reason"),
        F.current_timestamp().alias("computed_at"),
    )


def _batch_metrics(df: DataFrame):
    expressions = [F.count(F.lit(1)).alias("rows")]
    if "tick_count" in df.columns:
        expressions.append(F.coalesce(F.sum("tick_count"), F.lit(0)).alias("ticks"))
    else:
        expressions.append(F.lit(0).alias("ticks"))
    if "article_count" in df.columns:
        expressions.append(F.coalesce(F.sum("article_count"), F.lit(0)).alias("articles"))
        expressions.append(
            F.sum(F.when(F.col("article_count") > 0, 1).otherwise(0)).alias("rows_with_news")
        )
    else:
        expressions.append(F.lit(0).alias("articles"))
        expressions.append(F.lit(0).alias("rows_with_news"))
    row = df.agg(*expressions).first()
    return (
        int(row["rows"] or 0),
        int(row["ticks"] or 0),
        int(row["articles"] or 0),
        int(row["rows_with_news"] or 0),
    )


def _record_batch(run_id, batch_id, job_name, started_at, stats, status="SUCCESS", error=None):
    completed = datetime.now(timezone.utc)
    rows, ticks, articles, rows_with_news = stats
    gold_storage.write_audit_record({
        "run_id": run_id,
        "batch_id": int(batch_id),
        "job_name": job_name,
        "status": status,
        "input_tick_count": ticks,
        "input_news_count": articles,
        "output_row_count": rows,
        "rows_with_news": rows_with_news,
        "late_records": None,
        "started_at": started_at,
        "completed_at": completed,
        "duration_ms": int((completed - started_at).total_seconds() * 1000),
        "error_message": str(error)[:4000] if error else None,
    })


def _process_stock_batch(df: DataFrame, batch_id: int, run_id: str):
    started = datetime.now(timezone.utc)
    cached = df.persist()
    try:
        stats = _batch_metrics(cached)
        if stats[0]:
            gold_storage.upsert_dataframe(cached, "gold_stock_metrics")
        _record_batch(run_id, batch_id, "gold_v4_stock_metrics", started, stats)
        print(f"[GOLD V4][STOCK] batch={batch_id} rows={stats[0]} ticks={stats[1]}")
    except Exception as exc:
        try:
            _record_batch(run_id, batch_id, "gold_v4_stock_metrics", started, (0, 0, 0, 0), "FAILED", exc)
        finally:
            raise
    finally:
        cached.unpersist()


def _process_news_batch(df: DataFrame, batch_id: int, run_id: str):
    started = datetime.now(timezone.utc)
    cached = df.persist()
    try:
        stats = _batch_metrics(cached)
        if stats[0]:
            gold_storage.upsert_dataframe(cached, "gold_news_sentiment")
        _record_batch(run_id, batch_id, "gold_v4_news_articles", started, (stats[0], 0, stats[0], 0))
        print(f"[GOLD V4][NEWS] batch={batch_id} article-symbol rows={stats[0]}")
    except Exception as exc:
        try:
            _record_batch(run_id, batch_id, "gold_v4_news_articles", started, (0, 0, 0, 0), "FAILED", exc)
        finally:
            raise
    finally:
        cached.unpersist()


def _process_integrated_batch(df: DataFrame, batch_id: int, run_id: str):
    started = datetime.now(timezone.utc)
    cached = df.persist()
    try:
        stats = _batch_metrics(cached)
        if stats[0]:
            gold_storage.upsert_dataframe(cached, "gold_sentiment_price")

            alerts = (
                cached
                .where(F.col("alert_flag") == F.lit(True))
                .withColumn(
                    "alert_type",
                    F.when(
                        (F.col("avg_sentiment") <= settings.ALERT_SENTIMENT_THRESHOLD)
                        & (F.col("price_change_pct") <= settings.ALERT_PRICE_DROP_PCT),
                        F.lit("NEGATIVE_SENTIMENT_AND_PRICE_DROP"),
                    ).when(
                        F.col("avg_sentiment") <= settings.ALERT_SENTIMENT_THRESHOLD,
                        F.lit("NEGATIVE_SENTIMENT"),
                    ).otherwise(F.lit("PRICE_DROP")),
                )
                .select(
                    "symbol", "window_start", "alert_type",
                    F.col("alert_reason").alias("alert_message"),
                    F.col("avg_sentiment").alias("sentiment_score"),
                    "price_change_pct",
                    F.col("computed_at").alias("created_at"),
                )
            )
            gold_storage.upsert_dataframe(alerts, "gold_alerts")

        _record_batch(run_id, batch_id, "gold_v4_stock_news_join", started, stats)
        print(
            f"[GOLD V4][INTEGRATED] batch={batch_id} rows={stats[0]} "
            f"news_matches={stats[3]} articles={stats[2]}"
        )
    except Exception as exc:
        try:
            _record_batch(run_id, batch_id, "gold_v4_stock_news_join", started, (0, 0, 0, 0), "FAILED", exc)
        finally:
            raise
    finally:
        cached.unpersist()


def start_gold(spark, run_id: str):
    if not settings.PG_ENABLED:
        raise RuntimeError("FINTEL_PG_ENABLED must be 1 for Gold v4")
    if not settings.SILVER_TICKS_PATH.exists():
        raise FileNotFoundError(f"Stock Silver path not found: {settings.SILVER_TICKS_PATH}")
    if not settings.GOLD_V4_NEWS_INPUT_PATH.exists():
        settings.GOLD_V4_NEWS_INPUT_PATH.mkdir(parents=True, exist_ok=True)

    max_files = str(settings.GOLD_V4_MAX_FILES_PER_TRIGGER)

    ticks_raw = (
        spark.readStream
        .format("delta")
        .option("maxFilesPerTrigger", max_files)
        .load(str(settings.SILVER_TICKS_PATH))
    )
    news_raw = (
        spark.readStream
        .schema(NEWS_SCHEMA)
        .format("parquet")
        .option("maxFilesPerTrigger", max_files)
        .load(str(settings.GOLD_V4_NEWS_INPUT_PATH))
    )

    stock_events = _build_stock_events(ticks_raw)
    news_events = _build_news_events(news_raw)
    stock_windows = _build_stock_windows(stock_events)
    news_windows = _build_news_windows(news_events)
    stock_output = _stock_metrics_output(stock_windows)
    integrated_output = _build_integrated_stream(stock_windows, news_windows)

    checkpoint = settings.GOLD_V4_CHECKPOINT_PATH
    checkpoint.mkdir(parents=True, exist_ok=True)

    stock_run = f"{run_id}-stock"
    news_run = f"{run_id}-news"
    integrated_run = f"{run_id}-integrated"

    stock_query = (
        stock_output.writeStream
        .queryName("fintel-gold-v4-stock-metrics")
        .outputMode("append")
        .foreachBatch(lambda df, bid: _process_stock_batch(df, bid, stock_run))
        .option("checkpointLocation", str(checkpoint / "stock_metrics"))
        .trigger(processingTime=settings.GOLD_TRIGGER)
        .start()
    )

    news_output = news_events.withColumn("processed_at", F.current_timestamp()).select(
        "article_id", "symbol", "headline", "source", "published_at", "sentiment",
        "sentiment_score", "positive_score", "negative_score", "neutral_score", "processed_at",
    )
    news_query = (
        news_output.writeStream
        .queryName("fintel-gold-v4-news-articles")
        .outputMode("append")
        .foreachBatch(lambda df, bid: _process_news_batch(df, bid, news_run))
        .option("checkpointLocation", str(checkpoint / "news_articles"))
        .trigger(processingTime=settings.GOLD_TRIGGER)
        .start()
    )

    integrated_query = (
        integrated_output.writeStream
        .queryName("fintel-gold-v4-stock-news-join")
        .outputMode("append")
        .foreachBatch(lambda df, bid: _process_integrated_batch(df, bid, integrated_run))
        .option("checkpointLocation", str(checkpoint / "integrated"))
        .trigger(processingTime=settings.GOLD_TRIGGER)
        .start()
    )

    print("\n========== FINTEL GOLD V4 STARTED ==========")
    print("Stock Silver:", settings.SILVER_TICKS_PATH)
    print("News Silver :", settings.SILVER_NEWS_BY_TICKER_PATH)
    print("News inbox  :", settings.GOLD_V4_NEWS_INPUT_PATH)
    print("PostgreSQL  :", settings.PG_DB, "schema=", settings.PG_SCHEMA)
    print("Window      :", settings.WINDOW_DURATION)
    print("Slide       :", settings.SLIDE_DURATION)
    print("Watermark   :", settings.WATERMARK)
    print("Trigger     :", settings.GOLD_TRIGGER)
    print("Queries     : stock metrics / news articles / stream-stream integration")

    return [stock_query, news_query, integrated_query]
