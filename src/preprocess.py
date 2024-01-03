import pandas as pd
from sklearn.model_selection import train_test_split

ORIGINAL_DATA_FN = './data/original_data.csv'
TRAIN_DATA_FN = './data/train.csv'
TEST_DATA_FN = './data/test.csv'


def load_data():
    df = pd.read_csv(ORIGINAL_DATA_FN, index_col=[0])
    return df


def engineer_features(df):
    return df


def split_data(df):
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(TRAIN_DATA_FN, index=False)
    test.to_csv(TEST_DATA_FN, index=False)


if __name__ == '__main__':
    df = load_data()
    #split_data(df)