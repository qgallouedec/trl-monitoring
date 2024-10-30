from datasets import Dataset
from google.cloud import bigquery
import os


client = bigquery.Client()

query = """
#standardSQL
WITH daily_downloads AS (
  SELECT
    DATE(timestamp) AS day,
    COUNT(*) AS num_downloads
  FROM
    `bigquery-public-data.pypi.file_downloads`
  WHERE
    file.project = 'trl'
    -- Filter for the last 54 months
    AND DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 54 MONTH) AND CURRENT_DATE()
  GROUP BY
    day
)
SELECT
  day,
  num_downloads
FROM
  daily_downloads
ORDER BY
  day DESC
"""

query_job = client.query(query)
results = query_job.result()
df = results.to_dataframe()
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("qgallouedec/trl-metrics", config_name="pypi_downloads", token=os.environ.get("HF_TOKEN"))
