from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, avg, first
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import os, requests
import pygeohash as pgh
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reading environment variables for storage and API secrets
storage_account_key_input = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
storage_account_key_output = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY_TF")
key_opencage = os.environ.get("OPENCAGE_API_KEY")

# Azure Storage details
# Two different storage accounts are used: one for input data and another for output data
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
    """
    Retrieves latitude and longitude coordinates for a given address using the OpenCage Geocoding API.

    Args:
        address (str): The street address of the location.
        city (str): The city where the location is situated.
        country (str): The country of the location.

    Returns:
        tuple: A tuple containing the latitude and longitude (float, float).

    Notes:
    - The function constructs a full address from the given components.
    - It sends a GET request to the OpenCage API with proper retry handling.
    - Extracts and returns the latitude and longitude from the API response.
    """
    full_address = f"{address}, {city}, {country}"
    parameters = {"key": key_opencage, "q": full_address}

    session = requests.Session()
    #retry mechanism, retries 3 times with 0,5 sec backoff strategy
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get("https://api.opencagedata.com/geocode/v1/json", params=parameters)
    data = response.json()

    new_lat = data["results"][0]["geometry"]["lat"]
    new_lng = data["results"][0]["geometry"]["lng"]

    return new_lat, new_lng


# Defining schema for the geocode_udf, then creating the actual UDF for geocoding
geocode_schema = StructType([
    StructField("lat", DoubleType(), True),
    StructField("lng", DoubleType(), True)
])
geocode_udf = udf(opencage_geocode, geocode_schema)


def hotels_df_manipulation():
    """
    Cleans and updates the df_hotels DataFrame by filling in missing latitude and longitude values
    using the OpenCage Geocoding API.

    Steps:
    1. Identifies hotels with missing latitude or longitude values in the df_hotels.
    2. Calls the geocoding API to retrieve the correct coordinates and adds it to a new column named "geocode_result"
       in a new DataFrame called df_updated.
    3. Creates a temporary DataFrame(df_missing) with the "Id" column and the updated coordinates("new_lat", "new_lng").
    4. Performs a left join with the original DataFrame to integrate the new coordinates, using aliases.
    5. Updates the "Latitude" and "Longitude" columns, preserving existing values if they are not null, then removing the
       not needed columns("new_lat", "new_lng").

    Returns:
        A DataFrame with updated latitude and longitude values.
    """

    df_updated = df_hotels.withColumn(
        "geocode_result",
        when(
            (col("Latitude").cast("double").isNull()) | (col("Longitude").cast("double").isNull()),
            geocode_udf(col("Address"), col("City"), col("Country"))
        ).otherwise(None)
    )

    df_missing = df_updated.filter(
        (col("Latitude").cast("double").isNull()) | (col("Longitude").cast("double").isNull())
    ).select(
        col("Id"),
        col("geocode_result.lat").alias("new_lat"),
        col("geocode_result.lng").alias("new_lng")
    )

    df_orig = df_hotels.alias("orig")
    df_missing = df_missing.alias("upd")

    df_joined = df_orig.join(df_missing, on="Id", how="left")

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


def compute_geohash(lat, lon):
    """
    Computes a 4-character geohash for given latitude and longitude.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.

    Returns:
        str: Geohash of the given coordinates.
    """
    return pgh.encode(latitude=float(lat), longitude=float(lon), precision=4)

#Creating a string typed UDF called compute_geohash_udf which contains the compute_geohash function
compute_geohash_udf = udf(compute_geohash, StringType())

# Assigning geohash values by adding a new column and using the compute_geohash_udf, also casting the coordinate's
#columns to double
hotels_df_hashed = hotels_df_manipulation() \
    .withColumn("Latitude", col("Latitude").cast("double")) \
    .withColumn("Longitude", col("Longitude").cast("double")) \
    .withColumn("geohash", compute_geohash_udf(col("Latitude"), col("Longitude")))

weather_df_hashed = df_weather \
    .withColumn("lat", col("lat").cast("double")) \
    .withColumn("lng", col("lng").cast("double")) \
    .withColumn("geohash", compute_geohash_udf(col("lat"), col("lng")))

# Aggregating weather data grouped by date and the geohash
#The goal was to eliminate data multiplication by keeping only 1 geohash/day, with the average temperature of the many
#averages and keeping only the first coordinates of the coarse geohash
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

# Writing data into date-based directory structure with a year=month/day hierarchy,
# the date comes from the data's "wthr_date" column  e.g., "2016-10-01"
for row in date_groups:
    weather_date = row["wthr_date"]
    year, month, day = weather_date.split("-")
    output_path = f"{output_base_path}/year={year}/month={month}/day={day}"

    df_enriched_data.filter(col("wthr_date") == weather_date) \
        .write.parquet(output_path, mode="overwrite")

# Stopping Spark session
spark.stop()
