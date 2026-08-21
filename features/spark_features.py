"""Demonstrates the "20% of features awkward in SQL" pattern the plan calls
for: a per-applicant payment-trend slope (is this person's payment behavior
getting better or worse over time?), computed via a linear regression per
group. dbt/Postgres window functions can do rolling aggregates, but a
per-group regression slope needs either a custom aggregate or exactly this
kind of groupBy-apply — Spark is the natural tool once installments_payments
(13.6M rows) doesn't comfortably fit a single-machine loop.

Reads directly from the raw CSV rather than via JDBC — this is a batch
feature job, and reading from a data lake file is at least as realistic a
pattern as a live DB connection for this kind of workload, without needing a
JDBC driver bundled into the project just for a demo module.

PLATFORM NOTE: on native Windows Python, pandas_udf/Arrow-based operations
fail with a low-level Arrow IPC error (`WinError 10038`) — a known category
of fragility in PySpark's Arrow-based worker IPC on native Windows (plain
Spark operations like read/groupBy/count are unaffected). Runs correctly
in Linux — see `docker/spark_job.Dockerfile` for a working, verified
container (Java 8 + pyarrow==18.1.0; newer JVMs hit a separate Arrow
reflective-access restriction, and unpinned pyarrow hits a different
incompatibility — both diagnosed in PROGRESS.md Stage 22). Verified against
the real 13.6M-row installments_payments.csv: 339,587 applicants scored,
loaded into features.payment_trend_features.

Usage (Linux/WSL2, natively):
    python features/spark_features.py

Usage (from Windows, via the verified Docker container):
    docker build -t creditshield-spark-job -f docker/spark_job.Dockerfile .
    docker run --rm --network <compose-network> \
        -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 \
        -e POSTGRES_USER=pipeline -e POSTGRES_PASSWORD=pipeline123 -e POSTGRES_DB=credit_default \
        -v "$(pwd):/project" -w /project creditshield-spark-job

    Use the Compose service name (`postgres`), not the long
    `docker compose` container name — this project's directory name is long
    enough that `<project>-postgres-1` exceeds DNS's 63-character label
    limit and fails to resolve ("label too long"), a real bug hit and fixed
    during Stage 22, not a hypothetical caveat.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Must be set before importing pyspark.sql — Spark's Python workers spawn a
# subprocess using PYSPARK_PYTHON, and a stale copy of that env var on this
# machine (leftover from an unrelated conda environment) points at an
# interpreter that no longer exists. Pinning it to the running interpreter
# makes this reproducible regardless of what else is set in the host shell.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Requires Java 8 (see docker/spark_job.Dockerfile). Java 17+/21 hits an
# Arrow reflective-access restriction ("sun.misc.Unsafe or
# java.nio.DirectByteBuffer.<init>(long, int) not available") in pandas_udf
# that neither `spark.driver.extraJavaOptions` (SparkConf doesn't reach local
# mode's driver JVM — py4j launches it before Spark reads its config) nor
# `JDK_JAVA_OPTIONS` (confirmed via `java -version` to reach a bare JVM
# invocation, but not the one py4j spawns for Spark specifically) could work
# around — both real attempts, both dead ends, documented in PROGRESS.md
# Stage 22 rather than left as an unexplained "just use Java 8."

from pyspark.sql import SparkSession
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import DoubleType, StructField, StructType
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "installments_payments.csv"

OUTPUT_SCHEMA = StructType([
    StructField("sk_id_curr", DoubleType(), True),
    StructField("payment_trend_slope", DoubleType(), True),
])


def _slope_udf(df: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant: fit amt_payment ~ days_instalment, return the slope.
    Negative slope = payments shrinking over time (deteriorating ability to
    pay); positive = growing. Needs >= 3 points to fit meaningfully."""
    sk_id = df["sk_id_curr"].iloc[0]
    if len(df) < 3 or df["days_instalment"].nunique() < 2:
        return pd.DataFrame({"sk_id_curr": [sk_id], "payment_trend_slope": [0.0]})

    x = df["days_instalment"].values.astype(float)
    y = df["amt_payment"].fillna(0).values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0])
    return pd.DataFrame({"sk_id_curr": [sk_id], "payment_trend_slope": [slope]})


def compute_payment_trend_slopes(spark: SparkSession, csv_path: Path = DATA_PATH):
    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    df = df.select(
        df["SK_ID_CURR"].alias("sk_id_curr"),
        df["DAYS_INSTALMENT"].alias("days_instalment"),
        df["AMT_PAYMENT"].alias("amt_payment"),
    )

    slope_udf = pandas_udf(_slope_udf, OUTPUT_SCHEMA, PandasUDFType.GROUPED_MAP)
    return df.groupBy("sk_id_curr").apply(slope_udf)


def main():
    # The JVM available in this environment is a 32-bit HotSpot Client VM
    # (`java -version` reports "Client VM"; default MaxHeapSize ~256MB) —
    # 32-bit JVMs can't represent a 4g heap request at all ("Invalid maximum
    # heap size" at launch, not a runtime OOM). 1200m is close to what a
    # 32-bit process can actually address; 512m measurably OOM'd
    # (`OutOfMemoryError: Java heap space`) on the full 13.6M-row groupBy.
    # Smaller Arrow batches bound peak memory per pandas_udf call regardless
    # of how large any single applicant's group is.
    # local[*] (16 concurrent Python worker processes on this host) exhausted
    # system memory even after the JVM heap itself stopped OOMing — each
    # worker holds an Arrow-converted pandas batch in memory simultaneously.
    # local[4] trades parallelism for actually fitting in available RAM.
    spark = SparkSession.builder.appName("credit_default_spark_features") \
        .config("spark.driver.memory", "1200m") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5000") \
        .master("local[4]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        result = compute_payment_trend_slopes(spark)
        pdf = result.toPandas()
        print(f"Computed payment_trend_slope for {len(pdf)} applicants")

        url = (
            f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
            f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
        )
        engine = create_engine(url)
        pdf.to_sql("payment_trend_features", engine, schema="features", if_exists="replace", index=False)
        print("Loaded features.payment_trend_features")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
