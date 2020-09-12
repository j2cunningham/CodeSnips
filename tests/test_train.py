import sys
sys.path.append('/home/jeremy/gitlab/CodeSnips/train_model')
from train import setX, sety, readFile
import pandas as pd
import pytest

@pytest.fixture
def getDF():
    data = [['Alex',10],['Bob',12],['Clarke',13]]
    df = pd.DataFrame(data,columns=['Name','IsAdoptedUser'])
    return df

def test_fixture(getDF):
    assert type(getDF) != None

def test_setx(getDF):
    df = setX(getDF)
    assert df.shape == (3,1)

def test_sety(getDF):
    df = sety(getDF)
    assert df.name == 'IsAdoptedUser'

def test_readFile():
    tuple = readFile()
    assert len(tuple) == 2



# https://towardsdatascience.com/unit-testing-for-data-scientists-dc5e0cd397fb

