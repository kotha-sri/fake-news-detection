import pandas as pd
from anyascii import anyascii
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# welfake contains data from Kaggle, McIntire, Reuters, and Buzzfeed Political
welfake = pd.read_csv('WELFake_Dataset (1).csv')
welfake.drop(columns=['title'], inplace=True)
welfake.drop(welfake.columns[[0]], axis=1, inplace=True)

def remove_nonbinary(label):
    if label != 0 and label != 1:
        return None
    else:
        return label

welfake['label'].apply(remove_nonbinary)
welfake.dropna(axis=0, inplace=True)
welfake.drop_duplicates(inplace=True)

# preprocessing

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # anything in brackets
    text = re.sub("\\W"," ",text) # non-alphanumerics
    text = re.sub('https?://\S+|www\.\S+', '', text) # urls
    text = re.sub('<.*?>+', '', text) # html tags
    text = re.sub('[%s]' % re.escape('!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'), '', text) # punctuation
    text = re.sub('\n', '', text) # new lines
    text = re.sub('\w*\d\w*', '', text) # numbers within letters (not seperated by spaces)
    text = re.sub(' +', ' ', text) # multiple spaces
    text = re.sub('\%\r', '', text)
    text = anyascii(text)
    return text

welfake['text'] = welfake['text'].apply(wordopt)
x = welfake['text']
y = welfake['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectroizer = TfidfVectorizer(stop_words='english')
x_train = vectroizer.fit_transform(x_train)
x_test = vectroizer.transform(x_test)

# logistic regression model

LR = LogisticRegression()
LR.fit(x_train, y_train)

print(LR.score(x_test, y_test))