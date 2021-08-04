gs_bucket_name="churn-prediction-demo"
Bucket_uri="gs://churn-prediction-demo"
version=1
store_artifacts=Bucket_uri + "/" + str(version)
sample_data_path=Bucket_uri + "/" + "data/sample_churn_data.csv"
processed_data_path = Bucket_uri + "/" + "processed/processed_churn_data.csv"
