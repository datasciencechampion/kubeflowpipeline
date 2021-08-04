import global_config


# loading data into google cloud storage
from google.cloud import storage
storage_client = storage.Client()
#bucket.blob('data/sample_churn_data.csv').upload_from_filename('/home/jupyter/kubeflow-pipeline-demo/customer_churn_analysis/WA_Fn-UseC_-Telco-Customer-Churn.csv', content_type='text/csv')
bucket = storage_client.bucket(global_config.gs_bucket_name)
print("Loading samples into Cloud storage bucket : gs://churn-prediction-demo")
bucket.blob('data/sample_churn_data.csv').upload_from_filename('WA_Fn-UseC_-Telco-Customer-Churn.csv', content_type='text/csv')

print("Churn sample data loaded into Cloud storage bucket : gs://churn-prediction-demo")