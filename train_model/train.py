import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, anneal, Trials, space_eval
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle


from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import re
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

print("Current Time =", current_time)


def readFile():
    users_df = pd.read_csv('takehome_users.csv',encoding = 'latin-1')
    users_eng_df = pd.read_csv('takehome_user_engagement.csv',encoding = 'latin-1')
    return users_df, users_eng_df


def transform_users_df(users_df, user_Ids):
    users_df['IsInvited'] = users_df['invited_by_user_id'].apply(lambda x: 1 if x.is_integer() else 0)
    users_df['IsAdoptedUser'] = users_df['object_id'].apply(lambda x: 1 if (x in user_Ids) else 0)
    users_df['domain'] = users_df['email'].str.extract(r'(@[\w.]+)')
    doms = ['@gmail.com', '@yahoo.com', '@jourrapide.com', '@cuvox.de',
        '@gustr.com', '@hotmail.com']
    users_df['domain'] = users_df['domain'].apply(lambda x: x if (x in doms) else 'other')
    users_df['org_id'] = users_df['org_id'].apply(lambda x: x if (x < 11) else 99)
    users_df['org_id'].astype = 'object'

    users_df = users_df[['opted_in_to_mailing_list','enabled_for_marketing_drip','domain','creation_source','org_id','IsInvited','IsAdoptedUser']]
    users_df = pd.get_dummies(users_df,columns=['domain','creation_source','org_id'])
    return users_df

def transform_users_eng_df(users_eng_df):
    users_eng_df['shifted'] = users_eng_df.groupby(['user_id'])['time_stamp'].shift(2)
    users_eng_df['time_stamp']= pd.to_datetime(users_eng_df['time_stamp'])
    users_eng_df['shifted']=pd.to_datetime(users_eng_df['shifted'])
    users_eng_df['time_diff']=users_eng_df['time_stamp']-users_eng_df['shifted']
    users_eng_df = users_eng_df[users_eng_df['time_diff']<datetime.timedelta(days=3)]
    return users_eng_df

def get_user_ids(users_eng_df):
    user_Ids = users_eng_df.user_id.unique()
    return user_Ids

def setX(users_df):
    users_df.drop('IsAdoptedUser', axis=1, inplace=True)
    return users_df

def sety(users_df):
    y =  users_df['IsAdoptedUser']
    return y

def balance(X,y):
    # https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def splitUp(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    return X_train, X_test, y_train, y_test



def trainAndTune(X_train,y_train):
    
    n_iter=5
    random_state = 42
    num_folds = 3
    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)


    # possible values of parameters
    space={'n_estimators': hp.quniform('n_estimators', 100, 5000, 1),
        'max_depth' : hp.quniform('max_depth', 2, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 15, 1),
        'max_features': hp.quniform('max_features', 2, 5, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 10, 1)
        }


    def objective_func(params):
    # the function gets a set of variable parameters in "param"
        params = {'n_estimators': int(params['n_estimators']), 
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'min_samples_split': int(params['min_samples_split'])}
    
        # we use this params to create a classifier
        clf = RandomForestClassifier(random_state=random_state, **params)
        # and then conduct the cross validation with the same folds as before
        score = -cross_val_score(clf, X_train, y_train, cv=kf, scoring="precision", n_jobs=-1).mean()
        return score


    # trials will contain logging information
    trials = Trials()

    best=fmin(fn=objective_func, # function to optimize
            space=space, 
            algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
            max_evals=n_iter, # maximum number of iterations
            # trials=trials, # logging
            rstate=np.random.RandomState(random_state)) # fixing random state for the reproducibilit
            
    hyperparams = space_eval(space, best)
    print(hyperparams)
    return best

def model_fit(best):
    model_grid = RandomForestClassifier(random_state=42, n_estimators=int(best['n_estimators']),
            min_samples_leaf=int(best['min_samples_leaf']),
            max_features=int(best['max_features']),
            max_depth=int(best['max_depth']),min_samples_split=int(best['min_samples_split']))

    model_grid.fit(X_train,y_train)
    return model_grid

def pklDump(model):
    pickle.dump(model,open('rff.pkl','wb'))
# pickle.load( open( "rff.pkl", "rb" ) )


def main():
    files = readFile()
    users_df = files[0]
    users_eng_df = files[1]
    user_Ids = get_user_ids(users_eng_df)
    users_df_transformed = transform_users_df(users_df,user_Ids)
    users_eng_df_transformed = transform_users_eng_df(users_eng_df)
    y = sety(users_df_transformed)
    X = setX(users_df_transformed)
    balanced_data = balance(X,y)
    split_data = splitUp(balanced_data[0],balanced_data[1])
    tunedModel = trainAndTune(split_data[0],split_data[2])
    trainedModel = model_fit(tunedModel)
    pklDump(trainedModel)

if __name__ == '__main__':
    main()