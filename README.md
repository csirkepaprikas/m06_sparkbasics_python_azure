## Spark Basics Homework ##

Balázs Mikes

github link:
https://github.com/csirkepaprikas/m06_sparkbasics_python_azure.git
docker link:
https://hub.docker.com/repository/docker/michaelcorvin/last/general

This Spark Basic Homework is targeted to address several main topics of work with Spark.This project focuses on several key topics related to working with Spark. Through the completion of this homework I gained experience in various aspects of Spark development, including deploying a Spark Cluster to the cloud, developing Spark Jobs, and testing Spark applications.Additionally the goal was to become more familiar with important tools such as Terraform/Kubernetes, Azure cloud platforms, and Spark internals, such as developing basic logic within the Spark framework.

The actual task was to apply an ETL job – coded in python - on the source datas - saved on Azure Blob storage -: a set of hotels data in csv files and a set of weather datas in parquet format. The execution taoes place on Azure AKS. Both data sets contains longitude and latitude columns but in case of the hotels’ they might be missing or being saved in inappropiate format. The task was to clean this hotels’ data, fill with the proper coordinates by applying an API, then attach geohash to both of the data sources, then join and save them -also on Blob storage- in the same structured, partioned format as the source weather data.


# Here you can see the actual well-documented code:


'''python
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


      # I used this this dockerfile: 



#Modify this docker for your needs
FROM openjdk:8-alpine

ARG SPARK_VERSION_ARG=3.5.4
ARG HADOOP_VERSION=3.3.5

ENV BASE_IMAGE      openjdk:8-alpine
ENV SPARK_VERSION   $SPARK_VERSION_ARG

ENV SPARK_HOME      /opt/spark
ENV HADOOP_HOME     /opt/hadoop
ENV PATH            $PATH:$SPARK_HOME/bin
ENV SPARK_USER sparkuser
ENV SPARK_DRIVER_EXTRA_JAVA_OPTIONS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2"
ENV SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS="-Duser.home=/tmp -Divy.home=/tmp/.ivy2"
ENV IVY_HOME="/tmp/.ivy2"

RUN set -ex && \
    apk upgrade --no-cache && \
    apk --update add --no-cache bash tini libstdc++ glib gcompat libc6-compat linux-pam krb5 krb5-libs nss openssl wget sed curl && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
    # Removed the .cache to save space
    rm -rf /root/.cache && rm -rf /var/cache/apk/*

RUN wget -O /spark-${SPARK_VERSION}-bin-without-hadoop.tgz https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-without-hadoop.tgz && \
    tar -xzf /spark-${SPARK_VERSION}-bin-without-hadoop.tgz -C /opt/ && \
    ln -s /opt/spark-${SPARK_VERSION}-bin-without-hadoop $SPARK_HOME && \
    rm -f /spark-${SPARK_VERSION}-bin-without-hadoop.tgz && \
    mkdir -p $SPARK_HOME/work-dir && \
    mkdir -p $SPARK_HOME/spark-warehouse

RUN wget -O /hadoop-${HADOOP_VERSION}.tar.gz https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xzf /hadoop-${HADOOP_VERSION}.tar.gz -C /opt/ && \
    ln -s /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm -f /hadoop-${HADOOP_VERSION}.tar.gz

ENV PATH="$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH"
ENV SPARK_DIST_CLASSPATH $HADOOP_HOME/etc/hadoop:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/yarn:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/tools/lib/*
ENV SPARK_CLASSPATH $HADOOP_HOME/etc/hadoop:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/yarn:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/tools/lib/*

RUN wget -O $SPARK_HOME/jars/hadoop-azure-${HADOOP_VERSION}.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure/${HADOOP_VERSION}/hadoop-azure-${HADOOP_VERSION}.jar
RUN wget -O $SPARK_HOME/jars/hadoop-azure-datalake-${HADOOP_VERSION}.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure-datalake/${HADOOP_VERSION}/hadoop-azure-datalake-${HADOOP_VERSION}.jar
RUN wget -O $SPARK_HOME/jars/azure-storage-7.0.0.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-storage/7.0.0/azure-storage-7.0.0.jar
RUN wget -O $SPARK_HOME/jars/azure-data-lake-store-sdk-2.3.6.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-data-lake-store-sdk/2.3.6/azure-data-lake-store-sdk-2.3.6.jar
RUN wget -O $SPARK_HOME/jars/azure-keyvault-core-1.0.0.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-keyvault-core/1.0.0/azure-keyvault-core-1.0.0.jar
RUN wget -O $SPARK_HOME/jars/azure-keyvault-core-1.0.0.jar https://repo1.maven.org/maven2/ch/hsr/geohash/1.4.0/geohash-1.4.0.jar

COPY ./docker/entrypoint.sh /opt/
COPY ./requirements.txt /opt/
#COPY ./dist/sparkbasics-*.egg /opt/
COPY final_formatted2.py /opt/spark-3.5.4-bin-without-hadoop/work-dir/final_formatted2.py

RUN chmod +x /opt/*.sh

RUN apk update && \
    apk add --no-cache python3 py3-pip && \
    pip3 install --upgrade pip setuptools pygeohash requests && \
	pip3 install -r /opt/requirements.txt && \
    # Removed the .cache to save space
    rm -rf /var/cache/apk/*

WORKDIR /opt/spark-3.5.4-bin-without-hadoop/work-dir
ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
ARG spark_uid=185
USER ${spark_uid}


# To create this docker image: 


c:\data_eng\GIT\m06_sparkbasics_python_azure>docker build -f ./docker/HW1_Dockerfile_final2 -t last:last .
[+] Building 10.5s (22/22) FINISHED                                                                                                                                                           docker:desktop-linux
 => [internal] load build definition from HW1_Dockerfile_final2                                                                                                                                               0.0s
 => => transferring dockerfile: 4.36kB                                                                                                                                                                        0.0s
 => [internal] load metadata for docker.io/library/openjdk:8-alpine                                                                                                                                           1.1s
 => [auth] library/openjdk:pull token for registry-1.docker.io                                                                                                                                                0.0s
 => [internal] load .dockerignore                                                                                                                                                                             0.0s
 => => transferring context: 2B                                                                                                                                                                               0.0s
 => [ 1/16] FROM docker.io/library/openjdk:8-alpine@sha256:94792824df2df33402f201713f932b58cb9de94a0cd524164a0f2283343547b3                                                                                   0.1s
 => => resolve docker.io/library/openjdk:8-alpine@sha256:94792824df2df33402f201713f932b58cb9de94a0cd524164a0f2283343547b3                                                                                     0.0s
 => [internal] load build context                                                                                                                                                                             0.0s
 => => transferring context: 142B                                                                                                                                                                             0.0s
 => CACHED [ 2/16] RUN set -ex &&     apk upgrade --no-cache &&     apk --update add --no-cache bash tini libstdc++ glib gcompat libc6-compat linux-pam krb5 krb5-libs nss openssl wget sed curl &&     rm /  0.0s
 => CACHED [ 3/16] RUN wget -O /spark-3.5.4-bin-without-hadoop.tgz https://archive.apache.org/dist/spark/spark-3.5.4/spark-3.5.4-bin-without-hadoop.tgz &&     tar -xzf /spark-3.5.4-bin-without-hadoop.tgz   0.0s
 => CACHED [ 4/16] RUN wget -O /hadoop-3.3.5.tar.gz https://archive.apache.org/dist/hadoop/common/hadoop-3.3.5/hadoop-3.3.5.tar.gz &&     tar -xzf /hadoop-3.3.5.tar.gz -C /opt/ &&     ln -s /opt/hadoop-3.  0.0s
 => CACHED [ 5/16] RUN wget -O /opt/spark/jars/hadoop-azure-3.3.5.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure/3.3.5/hadoop-azure-3.3.5.jar                                              0.0s
 => CACHED [ 6/16] RUN wget -O /opt/spark/jars/hadoop-azure-datalake-3.3.5.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure-datalake/3.3.5/hadoop-azure-datalake-3.3.5.jar                   0.0s
 => CACHED [ 7/16] RUN wget -O /opt/spark/jars/azure-storage-7.0.0.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-storage/7.0.0/azure-storage-7.0.0.jar                                         0.0s
 => CACHED [ 8/16] RUN wget -O /opt/spark/jars/azure-data-lake-store-sdk-2.3.6.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-data-lake-store-sdk/2.3.6/azure-data-lake-store-sdk-2.3.6.jar     0.0s
 => CACHED [ 9/16] RUN wget -O /opt/spark/jars/azure-keyvault-core-1.0.0.jar https://repo1.maven.org/maven2/com/microsoft/azure/azure-keyvault-core/1.0.0/azure-keyvault-core-1.0.0.jar                       0.0s
 => CACHED [10/16] RUN wget -O /opt/spark/jars/azure-keyvault-core-1.0.0.jar https://repo1.maven.org/maven2/ch/hsr/geohash/1.4.0/geohash-1.4.0.jar                                                            0.0s
 => CACHED [11/16] COPY ./docker/entrypoint.sh /opt/                                                                                                                                                          0.0s
 => CACHED [12/16] COPY ./requirements.txt /opt/                                                                                                                                                              0.0s
 => CACHED [13/16] COPY final_formatted2.py /opt/spark-3.5.4-bin-without-hadoop/work-dir/final_formatted2.py                                                                                                  0.0s
 => CACHED [14/16] RUN chmod +x /opt/*.sh                                                                                                                                                                     0.0s
 => CACHED [15/16] RUN apk update &&     apk add --no-cache python3 py3-pip &&     pip3 install --upgrade pip setuptools pygeohash requests &&  pip3 install -r /opt/requirements.txt &&     rm -rf /var/cac  0.0s
 => CACHED [16/16] WORKDIR /opt/spark-3.5.4-bin-without-hadoop/work-dir                                                                                                                                       0.0s
 => exporting to image                                                                                                                                                                                        8.7s
 => => exporting layers                                                                                                                                                                                       0.0s
 => => exporting manifest sha256:d91e5de7ede43998685c0fe916248492724c3d948b9671a9e1fdd8da6cfc17ec                                                                                                             0.0s
 => => exporting config sha256:03f564674102f05a825370073982f88579fa833729f749fac22af2ab6deb8773                                                                                                               0.0s
 => => exporting attestation manifest sha256:4bbc30975d265f4de872bda7d3f321708988b29cf13e4be7350c4da737c85936                                                                                                 0.1s
 => => exporting manifest list sha256:e75ca35efe912bc7096f891876e28c20946ec31a198373b7d36e2f6416435c18                                                                                                        0.0s
 => => naming to docker.io/library/last:last                                                                                                                                                                  0.0s
 => => unpacking to docker.io/library/last:last                                                                                                                                                               8.5s

/////////////////////////////////////////////////////////////////////////
## I created an Azure Blob storage, than a homework1 container, where uploaded the source datas: ##
/////////////////////////////////////////////////////////////////////////

![source](https://github.com/user-attachments/assets/acf0dd8e-7769-44c2-a75b-aa68e62e4b8e)
![source2](https://github.com/user-attachments/assets/92ef9db7-15b0-480a-ac3a-f6ba2b9077ae)

/////////////////////////////////////////////////////////////////////////
## I Infrastructure creation with terraform (fraction of the plan): ##
/////////////////////////////////////////////////////////////////////////
+ resource "azurerm_kubernetes_cluster" "bdcc" {
      + dns_prefix                          = "bdccdevelopment"
      + location                            = "westeurope"
      + name                                = "aks-development-westeurope"
      + node_os_upgrade_channel             = "NodeImage"
      + private_cluster_enabled             = false
      + private_cluster_public_fqdn_enabled = false
      + resource_group_name                 = "rg-development-westeurope"
      + role_based_access_control_enabled   = true
      + run_command_enabled                 = true
      + sku_tier                            = "Free"
      + support_plan                        = "KubernetesOfficial"
      + tags                                = {
          + "env"    = "development"
          + "region" = "global"
        }
      + workload_identity_enabled           = false
          + name                 = "default"
          + node_count           = 1
          + os_disk_type         = "Managed"
          + scale_down_mode      = "Delete"
          + type                 = "VirtualMachineScaleSets"
          + ultra_ssd_enabled    = false
          + vm_size              = "Standard_E4s_v3"
    Successful creation: Apply complete! Resources: 1 added, 0 changed, 0 destroyed.


# After the successful infra creation I checked the fresh and crispy resources:

![rg_west_eu](https://github.com/user-attachments/assets/48566f52-35a8-4061-935b-cf7e87c9927f)
![rg_sub](https://github.com/user-attachments/assets/5206d943-2f66-4937-8024-b3542cbd7217)



# The VM size was a bottleneck, the free tier limited the usable VCPU numbers in 4, so by the second infra provisioning I’ve found the most suitable type: E4s_v3, which has 32GB memory, with this one I could decrease the processing time with 60%. Because of the limitation of the free tier I could only use 1 vCPU for the Driver and 1 for the Executor – on 1 thread- my approach was to maximize the RAM usage. ##


![vm](https://github.com/user-attachments/assets/66225dd8-8d60-4ee9-8088-0aaf7a1df27d)


# Here you can see the logs of the pods and the AKS node: 


NAME                                             READY   STATUS    RESTARTS   AGE
azure-blob-storage-etl-ad1de495292ed9-exec-1   1/1     Running   0          12m
proj1-f4c50895288001-driver                    1/1     Running   0          12m
mikesbala90 [ ~ ]$ kubectl describe proj1-f4c50895891001-driver
mikesbala90 [ ~ ]$ kubectl describe pod proj1-f4c508928891001-driver
Name:             proj1-f4c50895288911-driver
Namespace:        default
Priority:         0
Service Account:  sparkuser
Node:             aks-default-39658022-vmss000000/10.224.0.4
Start Time:       Fri, 21 Feb 2025 12:43:47 +0000
Labels:           spark-app-name=proj1
                  spark-app-selector=spark-f00da4edf5494da44d
                  spark-role=driver
                  spark-version=3.5.4
Annotations:      <none>
Status:           Running
IP:               10.244.0.160
IPs:
  IP:  10.244.0.160
Containers:
  spark-kubernetes-driver:
    Args:
      driver
      --properties-file
      /opt/spark/conf/spark.properties
      --class
      org.apache.spark.deploy.PythonRunner
      local:///opt/spark-3.5.4-bin-without-hadoop/work-dir/final_formatted2.py
    State:          Running
      Started:      Fri, 21 Feb 2025 12:43:49 +0000
    Ready:          True
    Restart Count:  0
    Limits:
      memory:  5734Mi
    Requests:
      cpu:     1
      memory:  5734Mi
    Environment:
      SPARK_USER:                    mikes
      SPARK_APPLICATION_ID:          spark-f00da4edf5494c
      SPARK_DRIVER_BIND_ADDRESS:      (v1:status.podIP)
      PYSPARK_PYTHON:                python3
      PYSPARK_DRIVER_PYTHON:         python3
      SPARK_LOCAL_DIRS:              /var/data/sp
      SPARK_CONF_DIR:                /opt/spark/conf
    Mounts:
      /opt/spark/conf from spark-conf-volume-driver (rw)
      /var/data/spark-b37-45c6-8e884a from spark-local-dir-1 (rw)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-phn5z (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  spark-local-dir-1:
    Type:       EmptyDir (a temporary directory that shares a pod's lifetime)
    Medium:     
    SizeLimit:  <unset>
  spark-conf-volume-driver:
    Type:      ConfigMap (a volume populated by a ConfigMap)
    Name:      spark-drv-dc8b0190a-conf-map
    Optional:  false
  kube-api-access-phn5z:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/memory-pressure:NoSchedule op=Exists
                             node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type     Reason       Age   From               Message
  ----     ------       ----  ----               -------
  Normal   Scheduled    13m   default-scheduler  Successfully assigned default/proj1-f4c5088891001-driver to aks-default-39658022-vmss000000
  Warning  FailedMount  13m   kubelet            MountVolume.SetUp failed for volume "spark-conf-volume-driver" : configmap "spark-drv-dc8b06952889190a-conf-map" not found
  Normal   Pulling      13m   kubelet            Pulling image "michaelcorvin/last:last"
  Normal   Pulled       13m   kubelet            Successfully pulled image "michaelcorvin/last:last" in 817ms (817ms including waiting). Image size: 1754735589 bytes.
  Normal   Created      13m   kubelet            Created container spark-kubernetes-driver
  Normal   Started      13m   kubelet            Started container spark-kubernetes-driver
mikesbala90 [ ~ ]$ kubectl describe node
Name:               aks-default-39658022-vmss000000
Roles:              <none>
Labels:             agentpool=default
                    beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/instance-type=Standard_E4s_v3
                    beta.kubernetes.io/os=linux
                    failure-domain.beta.kubernetes.io/region=westeurope
                    failure-domain.beta.kubernetes.io/zone=0
                    kubernetes.azure.com/agentpool=default
                    kubernetes.azure.com/azure-cni-overlay=true
                    kubernetes.azure.com/cluster=MC_rg-development-westeurope_aks-development-westeurope_westeur
                    kubernetes.azure.com/consolidated-additional-properties=40
                    kubernetes.azure.com/kubelet-identity-client-id=
                    kubernetes.azure.com/mode=system
                    kubernetes.azure.com/network-name=aks-vnet-12913004
                    kubernetes.azure.com/network-resourcegroup=rg-development-westeurope
                    kubernetes.azure.com/network-subnet=aks-subnet
                    kubernetes.azure.com/network-subscription=
                    kubernetes.azure.com/node-image-version=AKSUbuntu-2204gen2containerd-202502.03.0
                    kubernetes.azure.com/nodenetwork-vnetguid=8fd6372
                    kubernetes.azure.com/nodepool-type=VirtualMachineScaleSets
                    kubernetes.azure.com/os-sku=Ubuntu
                    kubernetes.azure.com/podnetwork-type=overlay
                    kubernetes.azure.com/role=agent
                    kubernetes.azure.com/storageprofile=managed
                    kubernetes.azure.com/storagetier=Premium_LRS
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=aks-default-3022-vmss000000
                    kubernetes.io/os=linux
                    node.kubernetes.io/instance-type=Standard_E4s_v3
                    storageprofile=managed
                    storagetier=Premium_LRS
                    topology.disk.csi.azure.com/zone=
                    topology.kubernetes.io/region=westeurope
                    topology.kubernetes.io/zone=0
Annotations:        alpha.kubernetes.io/provided-node-ip: 10.224.0.4
                    csi.volume.kubernetes.io/nodeid:
                      {"disk.csi.azure.com":"aks-default-3022-vmss000000","file.csi.azure.com":"aks-default-39658022-vmss000000"}
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Fri, 21 Feb 2025 09:59:56 +0000
Taints:             <none>
Unschedulable:      false
Lease:
  HolderIdentity:  aks-default-39658022-vmss000000
  AcquireTime:     <unset>
  RenewTime:       Fri, 21 Feb 2025 12:57:07 +0000
Conditions:
  Type                          Status  LastHeartbeatTime                 LastTransitionTime                Reason                          Message
  ----                          ------  -----------------                 ------------------                ------                          -------
  KernelDeadlock                False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   KernelHasNoDeadlock             kernel has no deadlock
  ReadonlyFilesystem            False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   FilesystemIsNotReadOnly         Filesystem is not read-only
  FrequentKubeletRestart        False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   NoFrequentKubeletRestart        kubelet is functioning properly
  VMEventScheduled              False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:36 +0000   NoVMEventScheduled              VM has no scheduled event
  FilesystemCorruptionProblem   False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   FilesystemIsOK                  Filesystem is healthy
  FrequentUnregisterNetDevice   False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   NoFrequentUnregisterNetDevice   node is functioning properly
  ContainerRuntimeProblem       False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   ContainerRuntimeIsUp            container runtime service is up
  KubeletProblem                False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   KubeletIsUp                     kubelet service is up
  FrequentDockerRestart         False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   NoFrequentDockerRestart         docker is functioning properly
  FrequentContainerdRestart     False   Fri, 21 Feb 2025 12:52:55 +0000   Fri, 21 Feb 2025 10:02:12 +0000   NoFrequentContainerdRestart     containerd is functioning properly
  MemoryPressure                False   Fri, 21 Feb 2025 12:55:51 +0000   Fri, 21 Feb 2025 09:59:56 +0000   KubeletHasSufficientMemory      kubelet has sufficient memory available
  DiskPressure                  False   Fri, 21 Feb 2025 12:55:51 +0000   Fri, 21 Feb 2025 09:59:56 +0000   KubeletHasNoDiskPressure        kubelet has no disk pressure
  PIDPressure                   False   Fri, 21 Feb 2025 12:55:51 +0000   Fri, 21 Feb 2025 09:59:56 +0000   KubeletHasSufficientPID         kubelet has sufficient PID available
  Ready                         True    Fri, 21 Feb 2025 12:55:51 +0000   Fri, 21 Feb 2025 10:00:11 +0000   KubeletReady                    kubelet is posting ready status
Addresses:
  InternalIP:  10.224.0.4
  Hostname:    aks-default-39658022-vmss000000
Capacity:
  cpu:                4
  ephemeral-storage:  129886128Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             32865016Ki
  pods:               250
Allocatable:
  cpu:                3860m
  ephemeral-storage:  119703055367
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             27591416Ki
  pods:               250
System Info:
  Machine ID:                 b64398ccc78b
  System UUID:                8ceed770-7c97-
  Boot ID:                    97aeee1a-cfdd-45d7-9bbe-4
  Kernel Version:             5.15.0-1079-azure
  OS Image:                   Ubuntu 22.04.5 LTS
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  containerd://1.7.25-1
  Kubelet Version:            v1.30.7
  Kube-Proxy Version:         v1.30.7
ProviderID:                   azure:///subscriptions/resourceGroups/mc_rg-development-westeurope_aks-development-westeurope_westeurope/providers/Microsoft.Compute/virtualMachineScaleSets/aks-default-39658022-vmss/virtualMachines/0
Non-terminated Pods:          (15 in total)
  Namespace                   Name                                              CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                                              ------------  ----------  ---------------  -------------  ---
  default                     azure-blob-storage-etl-ad1de49528892ed9-exec-1    1 (25%)       0 (0%)      20070Mi (74%)    20070Mi (74%)  13m
  default                     proj1-f4c5089528891001-driver                     1 (25%)       0 (0%)      5734Mi (21%)     5734Mi (21%)   13m
  kube-system                 azure-cns-s7fz9                                   40m (1%)      40m (1%)    250Mi (0%)       250Mi (0%)     177m
  kube-system                 azure-ip-masq-agent-hq24g                         100m (2%)     500m (12%)  50Mi (0%)        250Mi (0%)     177m
  kube-system                 cloud-node-manager-bbw6k                          50m (1%)      0 (0%)      50Mi (0%)        512Mi (1%)     177m
  kube-system                 coredns-659fcb469c-4xknh                          100m (2%)     3 (77%)     70Mi (0%)        500Mi (1%)     176m
  kube-system                 coredns-659fcb469c-jqtnd                          100m (2%)     3 (77%)     70Mi (0%)        500Mi (1%)     177m
  kube-system                 coredns-autoscaler-bfcb7c74c-kgh79                20m (0%)      200m (5%)   10Mi (0%)        500Mi (1%)     177m
  kube-system                 csi-azuredisk-node-sbcxn                          30m (0%)      0 (0%)      60Mi (0%)        1400Mi (5%)    177m
  kube-system                 csi-azurefile-node-cxhj2                          30m (0%)      0 (0%)      60Mi (0%)        600Mi (2%)     177m
  kube-system                 konnectivity-agent-67b5dc847f-fv2d7               20m (0%)      1 (25%)     20Mi (0%)        1Gi (3%)       176m
  kube-system                 konnectivity-agent-67b5dc847f-x7zpj               20m (0%)      1 (25%)     20Mi (0%)        1Gi (3%)       176m
  kube-system                 kube-proxy-fvm7t                                  100m (2%)     0 (0%)      0 (0%)           0 (0%)         177m
  kube-system                 metrics-server-5dfc656944-j72gw                   156m (4%)     251m (6%)   134Mi (0%)       404Mi (1%)     176m
  kube-system                 metrics-server-5dfc656944-tkq79                   156m (4%)     251m (6%)   134Mi (0%)       404Mi (1%)     176m
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests       Limits
  --------           --------       ------
  cpu                2922m (75%)    9242m (239%)
  memory             26732Mi (99%)  33172Mi (123%)
  ephemeral-storage  0 (0%)         0 (0%)
  hugepages-1Gi      0 (0%)         0 (0%)
  hugepages-2Mi      0 (0%)         0 (0%)
Events:              <none>


mikesbala90 [ ~ ]$ kubectl get pods
NAME                            READY   STATUS      RESTARTS   AGE
proj1-f4c5089528891001-driver   0/1     Completed   0          73m
mikesbala90 [ ~ ]$ kubectl describe pod proj1-f4c5089501-driver
Name:             proj1-f4c5089528891001-driver
Namespace:        default
Priority:         0
Service Account:  sparkuser
Node:             aks-default-39658022-vmss000000/10.224.0.4
Start Time:       Fri, 21 Feb 2025 12:43:47 +0000
Labels:           spark-app-name=proj1
                  spark-app-selector=spark-f00da4ee9552da44d
                  spark-role=driver
                  spark-version=3.5.4
Annotations:      <none>
Status:           Succeeded
IP:               10.244.0.160
IPs:
  IP:  10.244.0.160
Containers:
  spark-kubernetes-driver:
    Args:
      driver
      --properties-file
      /opt/spark/conf/spark.properties
      --class
      org.apache.spark.deploy.PythonRunner
      local:///opt/spark-3.5.4-bin-without-hadoop/work-dir/final_formatted2.py
    State:          Terminated
      Reason:       Completed
      Exit Code:    0
      Started:      Fri, 21 Feb 2025 12:43:49 +0000
      Finished:     Fri, 21 Feb 2025 13:55:43 +0000
    Ready:          False
    Restart Count:  0
    Limits:
      memory:  5734Mi
    Requests:
      cpu:     1
      memory:  5734Mi
    Environment:
      SPARK_USER:                    mikes
      SPARK_APPLICATION_ID:          spark-f00da4edf5494cb
      SPARK_DRIVER_BIND_ADDRESS:      (v1:status.podIP)
      PYSPARK_PYTHON:                python3
      PYSPARK_DRIVER_PYTHON:         python3
      SPARK_LOCAL_DIRS:              /var/data/spark-b3a750f6-aaa
      SPARK_CONF_DIR:                /opt/spark/conf
    Mounts:
      /opt/spark/conf from spark-conf-volume-driver (rw)
      /var/data/spark-b3a750f6-cc2d684a from spark-local-dir-1 (rw)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-phn5z (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   False 
  Initialized                 True 
  Ready                       False 
  ContainersReady             False 
  PodScheduled                True 
Volumes:
  spark-local-dir-1:
    Type:       EmptyDir (a temporary directory that shares a pod's lifetime)
    Medium:     
    SizeLimit:  <unset>
  spark-conf-volume-driver:
    Type:      ConfigMap (a volume populated by a ConfigMap)
    Name:      spark-drv-dc8b06952a-conf-map
    Optional:  false
  kube-api-access-phn5z:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/memory-pressure:NoSchedule op=Exists
                             node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:                      <none>


# Fraction of the spark-submit and the container building 


![submit](https://github.com/user-attachments/assets/7bbbed4b-2d64-40c2-9ada-2b3c6aa5d475)


# I used proxy server for the Spark job execution. I also applied port-forwarding to being able to access the Spark UI:


![portfrwd](https://github.com/user-attachments/assets/cbcb3bae-014a-40f6-84be-5f2b5937ed32)



# Here is the UI of the Spark, where In the Spark UI during an ETL job, you can get detailed insights into the execution process and the resource utilization of the cluster. The Jobs tab displays the status, runtime, and any potential errors of the jobs being executed. The Stages view shows the details of different processing steps, including the execution graph and shuffle operations. The Storage tab provides information on the state of RDDs and DataFrames in cache, while the Executors tab shows the status of the nodes running in the cluster, including their memory and task distribution


![ui_1](https://github.com/user-attachments/assets/cf00e31b-8a46-4abc-b24d-6766dd793302)
![ui_exec](https://github.com/user-attachments/assets/4e6a8462-0a71-4181-89b4-b144227ce690)
![ui_env](https://github.com/user-attachments/assets/ff022281-7f0d-4b7d-be51-9701a1640199)
![ui_tl3](https://github.com/user-attachments/assets/ec48bcb9-cab6-49f7-95c6-029354d5770a)


# The process took more than an hour, after the successful operation the pod status changed from Running to Completed:


![compl](https://github.com/user-attachments/assets/8b526a18-0c92-4070-a3ae-55fffab78a53)


# The enriched data was successfully created in the predefined directory, with the same structure and file format as the weather datas:


![destin](https://github.com/user-attachments/assets/1909b88b-557d-438b-ad76-fa1370c6bb96)


# Here you can see the content of the enriched data:


![enriched](https://github.com/user-attachments/assets/fd0f371d-a83d-44a9-8db5-e8a4ed6211ff)






