s3 = boto3.client('s3') 
obj = s3.get_object(Bucket=bucket, Key=train_data_key) 
df = pd.read_csv(obj['Body'],nrows=2000,index_col=0)


df = pd.read_csv(input_data,header=None,chunksize = 150)
for data in df:
    df_local = data
    break
    
df_local.shape

