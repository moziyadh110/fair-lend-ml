FROM eclipse-temurin:8-jre-jammy
# Java 8 explicitly, not whatever Debian/apt's default JDK is — this is the
# one JVM version confirmed (via H2O and the working Windows run) not to hit
# Arrow's reflective sun.misc.Unsafe restriction, which every Java 17/21
# combination tried has hit regardless of --add-opens flags or pyarrow
# version. See PROGRESS.md Stage 22.
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python
RUN pip3 install --no-cache-dir pyspark==3.5.6 pandas numpy "pyarrow==18.1.0" python-dotenv sqlalchemy psycopg2-binary
WORKDIR /project
CMD ["python", "features/spark_features.py"]
