from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, avg, first
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import os, requests
import pygeohash as pgh
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reading environment variables
storage_account_key_input = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
storage_account_key_output = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY_TF")
key_opencage = os.environ.get("OPENCAGE_API_KEY")

# Azure Storage details
storage_account_name_input = os.environ.get("STORAGE_ACCOUNT_NAME_INPUT")
storage_account_name_output = os.environ.get("STORAGE_ACCOUNT_NAME_OUTPUT")
input_container_name = os.environ.get("CONTAINER_NAME_INPUT")
output_container_name = os.environ.get("CONTAINER_NAME_OUTPUT")

# Initializing SparkSession
spark = SparkSession.builder \
    .appName("Azure Blob Storage ETL") \
    .config(f"spark.hadoop.fs.azure.account.key.{storage_account_name_input}.blob.core.windows.net",
            storage_account_key_input) \
    .config(f"spark.hadoop.fs.azure.account.key.{storage_account_name_output}.blob.core.windows.net",
            storage_account_key_output) \
    .config("spark.hadoop.fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .getOrCreate()

# Data file paths
hotels_url = f"wasbs://{input_container_name}@{storage_account_name_input}.blob.core.windows.net/hotels/"
weather_url = f"wasbs://{input_container_name}@{storage_account_name_input}.blob.core.windows.net/weather/year=2016/"
parquet_output_path = f"wasbs://{input_container_name}@{storage_account_name_input}.blob.core.windows.net/hotels_parquet/"

# Reading CSV file and converting to Parquet format
df_hotels_csv = spark.read.csv(hotels_url, header=True, inferSchema=True)
df_hotels_csv.write.mode("overwrite").parquet(parquet_output_path)

# Reading Parquet files
df_hotels = spark.read.parquet(parquet_output_path)
df_weather = spark.read.parquet(weather_url)


# OpenCage API call for retrieving geographic coordinates
def opencage_geocode(address, city, country):
    full_address = f"{address}, {city}, {country}"
    parameters = {"key": key_opencage, "q": full_address}

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get("https://api.opencagedata.com/geocode/v1/json", params=parameters)
    data = response.json()

    new_lat = data["results"][0]["geometry"]["lat"]
    new_lng = data["results"][0]["geometry"]["lng"]

    return new_lat, new_lng


# Creating UDF for geocoding
geocode_schema = StructType([
    StructField("lat", DoubleType(), True),
    StructField("lng", DoubleType(), True)
])
geocode_udf = udf(opencage_geocode, geocode_schema)


# Data transformation to fill missing coordinates
def hotels_df_manipulation():
    df_updated = df_hotels.withColumn(
        "geocode_result",
        when(
            (col("Latitude").cast("double").isNull()) | (col("Longitude").cast("double").isNull()),
            geocode_udf(col("Address"), col("City"), col("Country"))
        ).otherwise(None)
    )

    # Creating a new DataFrame with missing coordinates
    df_missing = df_updated.filter(
        (col("Latitude").cast("double").isNull()) | (col("Longitude").cast("double").isNull())
    ).select(
        col("Id"),
        col("geocode_result.lat").alias("new_lat"),
        col("geocode_result.lng").alias("new_lng")
    )

    # Joining the original and updated DataFrames
    df_orig = df_hotels.alias("orig")
    df_missing = df_missing.alias("upd")

    df_joined = df_orig.join(df_missing, on="Id", how="left")

    # Updating coordinates
    df_final = df_joined.withColumn(
        "Latitude",
        when(col("orig.Latitude").cast("double").isNull(), col("upd.new_lat"))
        .otherwise(col("orig.Latitude"))
    ).withColumn(
        "Longitude",
        when(col("orig.Longitude").cast("double").isNull(), col("upd.new_lng"))
        .otherwise(col("orig.Longitude"))
    ).drop("new_lat", "new_lng")

    return df_final


# Geohash computation
def compute_geohash(lat, lon):
    return pgh.encode(latitude=float(lat), longitude=float(lon), precision=4)


compute_geohash_udf = udf(compute_geohash, StringType())

# Assigning geohash values
hotels_df_hashed = hotels_df_manipulation() \
    .withColumn("Latitude", col("Latitude").cast("double")) \
    .withColumn("Longitude", col("Longitude").cast("double")) \
    .withColumn("geohash", compute_geohash_udf(col("Latitude"), col("Longitude")))

weather_df_hashed = df_weather \
    .withColumn("lat", col("lat").cast("double")) \
    .withColumn("lng", col("lng").cast("double")) \
    .withColumn("geohash", compute_geohash_udf(col("lat"), col("lng")))

# Aggregating weather data
weather_df_aggregated = weather_df_hashed.groupBy("wthr_date", "geohash").agg(
    avg("avg_tmpr_f").alias("avg_tmpr_f"),
    avg("avg_tmpr_c").alias("avg_tmpr_c"),
    first("lat").alias("lat"),
    first("lng").alias("lng")
)

# Joining hotel and weather data
df_enriched_data = weather_df_aggregated.join(hotels_df_hashed, on="geohash", how="left")

# Defining the output path
output_base_path = f"wasbs://{output_container_name}@{storage_account_name_output}.blob.core.windows.net/"

# Collecting unique dates
date_groups = df_enriched_data.select("wthr_date").distinct().collect()

# Writing data into date-based directory structure
for row in date_groups:
    weather_date = row["wthr_date"]  # e.g., "2016-10-01"
    year, month, day = weather_date.split("-")
    output_path = f"{output_base_path}/year={year}/month={month}/day={day}"

    df_enriched_data.filter(col("wthr_date") == weather_date) \
        .write.parquet(output_path, mode="overwrite")

# Stopping Spark session
spark.stop()
