# Looking for missing values
missing = pd.concat([df.isnull().sum(), 100 * df.isnull().mean()],axis = 1)
missing.columns=['count', '%']
missing = missing.sort_values(by='count')
missing
