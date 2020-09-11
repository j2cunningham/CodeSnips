import sys
sys.path.append('/home/jeremy/gitlab/CodeSnips/train_model')
from train import setX
import pandas as pd
import pytest

def test_setx():
    data = [['Alex',10],['Bob',12],['Clarke',13]]
    df = pd.DataFrame(data,columns=['Name','IsAdoptedUser'])
    df = setX(df)
    print(df.shape)
    assert df.shape == (3,1)

# https://towardsdatascience.com/unit-testing-for-data-scientists-dc5e0cd397fb

