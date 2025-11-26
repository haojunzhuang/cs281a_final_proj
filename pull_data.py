"""
Usage: in terminal , run:

~/google-cloud-sdk/bin/gcloud init
~/google-cloud-sdk/bin/gcloud auth application-default login 

"""

from google.cloud import bigquery
client = bigquery.Client(project="physionet-479306")

tables = ["icu_cohort_small", "vitals_ts_small", "labs_ts_small"]
project = "physionet-479306"
dataset = "my_dataset"

for table in tables:
    query = f"SELECT * FROM `{project}.{dataset}.{table}`"
    df = client.query(query).to_dataframe()
    df.to_csv(f"data/{table}.csv", index=False)
    print(f"Downloaded: {table}.csv")