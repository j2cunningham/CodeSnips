s3 = boto3.client('s3') 
obj = s3.get_object(Bucket=bucket, Key=train_data_key) 
df = pd.read_csv(obj['Body'],nrows=2000,index_col=0)

