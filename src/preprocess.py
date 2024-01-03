import pandas as pd
from sklearn.model_selection import train_test_split

ORIGINAL_DATA_FN = './data/original_data.csv'
CURATED_DATA_FN = './data/curated_data.csv'


def engineer_features(df):
    # Extract hour from account creation timestamp
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour_created'] = [date.hour for date in df['created_at']]
    df = df.drop(columns='created_at')

    # Extract length of description (bio)
    df['description_length'] = [len(str(desc)) if not pd.isna(desc) 
                                else 0 for desc in df['description']]
    df = df.drop(columns='description')

    # Remove usefless features
    useless_features = ['id', 'profile_background_image_url', 'profile_image_url']
    df = df.drop(columns=useless_features)

    # One hot encode language
    df = pd.get_dummies(df, columns=['lang'])

    # Determine whether location is given
    df['location'] = [True if loc!='unknown' else False for loc in df['location']]

    # Extract length of screen name
    df['name_length'] = [len(name) for name in df['screen_name']]
    df = df.drop(columns='screen_name')

    return df


if __name__ == '__main__':
    # Engineer features for train and test
    df = pd.read_csv(ORIGINAL_DATA_FN, index_col=[0])
    df = engineer_features(df)
    df.to_csv(CURATED_DATA_FN, index=False)