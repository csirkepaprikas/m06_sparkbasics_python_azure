import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, avg, first
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import os, requests
import pygeohash as pgh
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

key_opencage = os.environ.get("OPENCAGE_API_KEY")

# Dummy datas
hotels_data = [
    (1, "Hotel A", "Kossuth Lajos tér 1", "Budapest", "Hungary", 47.4979, 19.0402),
    (2, "Hotel B", "Andrássy út 10", "Budapest", "Hungary", None, None),
]

weather_data = [
    ("2025-02-23", 47.4979, 19.0402, 15.5, 10.5),
    ("2025-02-23", 47.4979, 19.0402, 14.0, 9.0),
]


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.master("local[1]").appName("HotelWeatherTests").getOrCreate()
    yield spark
    spark.stop()


def compute_geohash(lat, lon):
    return pgh.encode(latitude=float(lat), longitude=float(lon), precision=4)

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


def test_compute_geohash():
    lat, lon = 47.4979, 19.0402  # Budapest coordinates
    geohash = compute_geohash(lat, lon)

    assert isinstance(geohash, str), "Geohash should be a string"
    assert len(geohash) == 4, "Geohash should be 4 characters long"


def test_opencage_geocode():
    # We are using a real address, the API must respond with the same coordinates
    latitude, longitude = opencage_geocode("Kossuth Lajos tér 1", "Budapest", "Hungary")

    assert 47.49 < latitude < 47.51
    assert 19.03 < longitude < 19.05


def test_hotels_df_manipulation(spark):
    schema = ["Id", "Name", "Address", "City", "Country", "Latitude", "Longitude"]
    df_hotels = spark.createDataFrame(hotels_data, schema=schema)

    # UDF for geocoding
    geocode_udf = udf(opencage_geocode, returnType=StructType([
        StructField("lat", DoubleType(), True),
        StructField("lng", DoubleType(), True)
    ]))

    #Adding a new "geocode_result" column with generated coordinates
    df_updated = df_hotels.withColumn(
        "geocode_result",
        when(
            (col("Latitude").cast("double").isNull()) | (col("Longitude").cast("double").isNull()),
            geocode_udf(col("Address"), col("City"), col("Country"))
        ).otherwise(None)
    )
    #Creating a new DF for those rows where the coordinates missing, the "Id" and the separated coordinates are defined
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

    # Updating coordinates, where needed, then the deleting the unnecessary "new_lat" and "new_lng" columns
    df_hotel_final = df_joined.withColumn(
        "Latitude",
        when(col("orig.Latitude").cast("double").isNull(), col("upd.new_lat"))
        .otherwise(col("orig.Latitude"))
    ).withColumn(
        "Longitude",
        when(col("orig.Longitude").cast("double").isNull(), col("upd.new_lng"))
        .otherwise(col("orig.Longitude"))
    ).drop("new_lat", "new_lng")
    #Count the rows which were filtered due ti wrong format (can't be cast to double).
    assert df_hotel_final.filter(col("Latitude").cast("double").isNull() | col("Longitude").cast("double").isNull()).count() == 0
    return df_hotel_final




def test_join_weather_hotels(spark):
    #schema_hotels = ["Id", "Name", "Address", "City", "Country", "Latitude", "Longitude"]
    schema_weather = ["wthr_date", "lat", "lng", "avg_tmpr_f", "avg_tmpr_c"]

    #df_hotels = spark.createDataFrame(hotels_data, schema_hotels)
    df_weather = spark.createDataFrame(weather_data, schema_weather)

    # Geohash UDF
    compute_geohash_udf = udf(compute_geohash, StringType())

    # Geohash computing
    df_hotels = test_hotels_df_manipulation(spark).withColumn("geohash", compute_geohash_udf(col("Latitude"), col("Longitude")))
    df_weather = df_weather.withColumn("geohash", compute_geohash_udf(col("lat"), col("lng")))

    # Joining df_weather and df_hotels DataFrames
    df_joined = df_weather.join(df_hotels, on="geohash", how="left")

    # Validating that 4 rows are joined
    assert df_joined.count() == 4

    # Validating the number of columns
    assert len(df_joined.columns) == 13

    # Validating the column names
    expected_columns = [
        "wthr_date", "lat", "lng", "avg_tmpr_f", "avg_tmpr_c",
        "Id", "Name", "Address", "City", "Country", "Latitude", "Longitude", "geohash"
    ]
    assert set(df_joined.columns) == set(expected_columns)
