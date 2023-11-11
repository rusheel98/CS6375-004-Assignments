import re
import pandas as pd
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def remove_hashtags(tweet):
    return tweet.replace("#", "")


def remove_mentions(tweet):
    return re.sub(r'@\w+', '', tweet)


def remove_urls(tweet):
    return re.sub(r'http\S+', '', tweet)


def to_lowercase(tweet):
    return tweet.lower()


class Preprocess:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _tokenize_and_lemmatize(self, tweet):
        tokens = word_tokenize(tweet)
        return [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]

    def _remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def _process_tweet(self, tweet: str) -> Set[str]:
        tweet = remove_mentions(tweet)
        tweet = remove_hashtags(tweet)
        tweet = remove_urls(tweet)
        tweet = to_lowercase(tweet)
        tokens = self._tokenize_and_lemmatize(tweet)
        tokens = self._remove_stop_words(tokens)
        return set(tokens)

    def __call__(self, _tweets: pd.DataFrame) -> pd.DataFrame:
        df = _tweets.copy()
        df['tokens'] = df['tweet'].apply(self._process_tweet)
        return df


if __name__ == "__main__":
    # test preprocessing
    tweets = pd.read_csv(
        "https://raw.githubusercontent.com/chaitanya-basava/CS6375-004-Assignment-1-data/main/bbchealth.txt",
        names=['id', 'datetime', 'tweet'], sep='|'
    )

    preprocessor = Preprocess()
    preprocessed_tweets = preprocessor(tweets)

    print(preprocessed_tweets.head(10).to_string(index=False))
