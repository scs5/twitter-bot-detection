import pandas as pd
import numpy as np
from afinn import Afinn
from utils import count_digits, count_pos_tags
import nltk
import emoji
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

ORIGINAL_DATA_FN = './data/original_data.csv'
CURATED_DATA_FN = './data/curated_data.csv'


def engineer_features(df):
    """
    Features to implement:
        - Consider other languages when POS tagging
    """
    # Handle NaN values
    mode = df['description'].mode().values[0]
    df['description'].fillna(value=mode, inplace=True)

    # Label encode class (account type)
    label_encoder = LabelEncoder()
    df['account_type'] = label_encoder.fit_transform(df['account_type'])

    # Extract hour from account creation timestamp
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour_created'] = [date.hour for date in df['created_at']]

    # Number of characters in description
    df['description_chars'] = [len(str(desc)) if not pd.isna(desc) 
                                else 0 for desc in df['description']]
    
    # Number of words in description
    df['description_words'] = [len(str(desc).split()) if not pd.isna(desc) 
                                else 0 for desc in df['description']]

    # One hot encode language
    df = pd.get_dummies(df, columns=['lang'])

    # Determine whether location is given
    df['location'] = [True if loc!='unknown' else False for loc in df['location']]

    # Extract length of screen name
    df['name_length'] = [len(name) for name in df['screen_name']]

    # Number of digits in screen name
    df['num_digits_in_name'] = [count_digits(name) for name in df['screen_name']]

    # POS frequency in profile description
    df_pos_tags = df['description'].apply(count_pos_tags).apply(pd.Series)
    df = pd.concat([df, df_pos_tags], axis=1)
        
    # Sentiment score of description
    afn = Afinn()
    df['sentiment'] = [afn.score(desc) for desc in df['description']]

    # Number of emojis in description
    df['num_emojis'] = [emoji.emoji_count(desc) for desc in df['description']]

    # Remove useless features
    useless_features = ['id', 'profile_background_image_url', 'profile_image_url']
    df = df.drop(columns=useless_features)
    df = df.drop(columns='screen_name')
    df = df.drop(columns='created_at')
    df = df.drop(columns='description')

    return df


if __name__ == '__main__':
    # Engineer features
    df = pd.read_csv(ORIGINAL_DATA_FN, index_col=[0])
    df = engineer_features(df)
    df.to_csv(CURATED_DATA_FN, index=False)