import pandas as pd
import global_config

#gs_bucket_name="churn-prediction-demo"
#Bucket_uri="gs://churn-prediction-demo"
#version=1
#store_artifacts=Bucket_uri + "/" + str(version)
#sample_data_path=Bucket_uri + "/" + "data/sample_churn_data.csv"
#processed_data_path=Bucket_uri + "/" + "processed/churn_processed_data.csv"

df = pd.read_csv(global_config.sample_data_path)
#df = pd.read_csv(sample_data_path)

#data Cleaning
#the TotalCharges is the amount charged to the customer, hence this should be numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# SeniorCitizen should be a qualitative column and not numeric, hence converting it into a object type
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')

#Fill null values
df['TotalCharges'].fillna(value=df['tenure'] * df['MonthlyCharges'], inplace=True)

#convert churn class to numeric
'''
Tensorflow requires a Boolean value to train the classifier. We need to convert the values from string to integer. The label is store as an object, however, we need to convert it into a numeric value. 
'''
def churn_to_numeric(value):
    if value.lower() == 'yes':
        return 1
    return 0
df['Churn'] = df['Churn'].apply(churn_to_numeric)

print("Data preprocessing done Sucessfully")

df.to_csv("processed_data.csv")
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(global_config.gs_bucket_name)
bucket.blob('processed/processed_churn_data.csv').upload_from_filename('processed_data.csv', content_type='text/csv')
print("Processed Churn data loaded Sucessfully into Cloud storage bucket : gs://churn-prediction-demo")

