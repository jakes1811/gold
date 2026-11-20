"""Read/write helpers that hide the Delta-vs-Parquet switch (settings.TABLE_FORMAT)."""
from config import settings


def read_stream(spark, path, schema_ddl: str = None, fmt: str = None, options: dict = None):
    """Stream a table folder. Parquet streaming needs an explicit schema; Delta does not."""
    fmt = fmt or settings.TABLE_FORMAT
    reader = spark.readStream.format(fmt)
    for key, value in (options or {}).items():
        reader = reader.option(key, value)
    if fmt != "delta" and schema_ddl:
        reader = reader.schema(schema_ddl)
    return reader.load(str(path))


def write_batch(df, path, mode: str = "append", app_id: str = None, batch_id: int = None, fmt: str = None):
    """Idempotent micro-batch write (Delta txnAppId/txnVersion when Delta)."""
    fmt = fmt or settings.TABLE_FORMAT
    writer = df.write.format(fmt).mode(mode)
    if fmt == "delta" and app_id is not None and batch_id is not None:
        writer = writer.option("txnAppId", app_id).option("txnVersion", batch_id)
    writer.save(str(path))
