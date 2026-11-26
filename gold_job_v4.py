"""FinTel Gold v4.

Three independent Structured Streaming queries:
  1. Stock windows -> gold_stock_metrics.
  2. News articles -> gold_news_sentiment (one row per article-symbol).
  3. Stock windows enriched with the latest available news -> gold_sentiment_price.

Integration is a batch-time as-of lookup against PostgreSQL. For every stock
window, Gold selects the latest news article for the same symbol whose
event_time is <= the stock window end. News is carried forward indefinitely
until a newer article arrives. No exact window equality or news expiry is used.

Gold v4 writes directly to PostgreSQL and uses its isolated v4 checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import traceback

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
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
    """
    Enrich each completed stock window with the latest same-symbol news
    available by that window's end time. The news lookup is a PostgreSQL
    static/batch lookup, not a stream-stream window join.
    """
    started = datetime.now(timezone.utc)
    cached = df.persist()
    stage = "start"
    LOGGER.info("INTEGRATED batch=%s: started", batch_id)
    try:
        LOGGER.info("INTEGRATED batch=%s: collecting stock window rows", batch_id)
        stock_rows = cached.collect()
        stage = "stock_rows_collected"
        LOGGER.info("INTEGRATED batch=%s: collected stock_rows=%s", batch_id, len(stock_rows))
        if not stock_rows:
            stats = (0, 0, 0, 0)
            _record_batch(run_id, batch_id, "gold_v4_stock_news_asof", started, stats)
            print(f"[GOLD V4][INTEGRATED] batch={batch_id} rows=0")
            return

        stage = "fetch_latest_news"
        latest_news = gold_storage.fetch_latest_news_for_windows(stock_rows)
        LOGGER.info("INTEGRATED batch=%s: fetched latest_news=%s", batch_id, len(latest_news))

        news_by_key = {
            (item["symbol"], item["window_start"]): item
            for item in latest_news
        }

        stage = "enriching_rows"
        enriched = []
        for row in stock_rows:
            record = row.asDict(recursive=True)
            news = news_by_key.get((record["symbol"], record["window_start"]), {})

            article_id = news.get("article_id")
            sentiment_score = news.get("avg_sentiment")
            price_change = record.get("price_change_pct")

            negative_sentiment = (
                sentiment_score is not None
                and sentiment_score <= settings.ALERT_SENTIMENT_THRESHOLD
            )
            price_drop = (
                price_change is not None
                and price_change <= settings.ALERT_PRICE_DROP_PCT
            )
            alert_flag = bool(negative_sentiment or price_drop)

            if negative_sentiment and price_drop:
                alert_reason = (
                    f"Negative sentiment and price drop; sentiment={sentiment_score:.3f}; "
                    f"price change={price_change:.3f}%"
                )
            elif negative_sentiment:
                alert_reason = f"Negative news sentiment: {sentiment_score:.3f}"
            elif price_drop:
                alert_reason = f"Price drop: {price_change:.3f}%"
            else:
                alert_reason = None

            record.update({
                "article_count": 1 if article_id is not None else 0,
                "avg_sentiment": news.get("avg_sentiment"),
                "min_sentiment": news.get("min_sentiment"),
                "latest_headline": news.get("latest_headline"),
                "latest_sentiment_label": news.get("latest_sentiment_label"),
                "latest_news_time": news.get("latest_news_time"),
                "alert_flag": alert_flag,
                "alert_reason": alert_reason,
                "computed_at": datetime.now(timezone.utc),
            })
            enriched.append(record)

        LOGGER.info("INTEGRATED batch=%s: enrichment complete enriched_rows=%s", batch_id, len(enriched))

        enriched_schema = StructType(cached.schema.fields + [
            StructField("article_count", IntegerType(), False),
            StructField("avg_sentiment", DoubleType(), True),
            StructField("min_sentiment", DoubleType(), True),
            StructField("latest_headline", StringType(), True),
            StructField("latest_sentiment_label", StringType(), True),
            StructField("latest_news_time", TimestampType(), True),
            StructField("alert_flag", BooleanType(), False),
            StructField("alert_reason", StringType(), True),
        ])
        stage = "create_integrated_dataframe"
        integrated = cached.sparkSession.createDataFrame(enriched, schema=enriched_schema)
        LOGGER.info("INTEGRATED batch=%s: integrated DataFrame created", batch_id)

        stage = "batch_metrics"
        stats = _batch_metrics(integrated)
        LOGGER.info("INTEGRATED batch=%s: batch metrics=%s", batch_id, stats)
        stage = "write_sentiment_price"
        gold_storage.upsert_dataframe(integrated, "gold_sentiment_price")
        LOGGER.info("INTEGRATED batch=%s: gold_sentiment_price written", batch_id)

        stage = "building_alerts"
        alerts = (
            integrated
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
        LOGGER.info("INTEGRATED batch=%s: alert DataFrame constructed", batch_id)
        gold_storage.upsert_dataframe(alerts, "gold_alerts")
        LOGGER.info("INTEGRATED batch=%s: gold_alerts write completed", batch_id)

        _record_batch(run_id, batch_id, "gold_v4_stock_news_asof", started, stats)
        LOGGER.info("INTEGRATED batch=%s: completed rows=%s", batch_id, stats[0])
        print(
            f"[GOLD V4][INTEGRATED] batch={batch_id} rows={stats[0]} "
            f"windows_with_news={stats[3]} latest-news-as-of=enabled"
        )
    except Exception as exc:
        LOGGER.error("INTEGRATED batch=%s FAILED at stage=%s: %s", batch_id, stage, exc)
        LOGGER.error("INTEGRATED batch=%s traceback:\n%s", batch_id, traceback.format_exc())
        try:
            _record_batch(
                run_id, batch_id, "gold_v4_stock_news_asof",
                started, (0, 0, 0, 0), "FAILED", exc
            )
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
    stock_output = _stock_metrics_output(stock_windows)

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
        "article_id", "symbol", "headline", "source", "published_at", "event_time",
        "sentiment", "sentiment_score", "positive_score", "negative_score",
        "neutral_score", "processed_at",
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
        stock_output.writeStream
        .queryName("fintel-gold-v4-stock-news-asof")
        .outputMode("append")
        .foreachBatch(lambda df, bid: _process_integrated_batch(df, bid, integrated_run))
        .option("checkpointLocation", str(checkpoint / "integrated_v2"))
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
    print("Queries     : stock metrics / news articles / PostgreSQL as-of enrichment")

    return [stock_query, news_query, integrated_query]

LOGGER = logging.getLogger("gold_v4.integrated")
