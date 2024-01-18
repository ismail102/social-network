from nltk.corpus import stopwords
from re import sub
import string
import neattext.functions as nfx

def remove_usernames(raw):
        return sub(r'@[^\s]*', '', raw)

# @staticmethod
def remove_punctuation(raw):
    return raw.translate(str.maketrans('', '', string.punctuation))

# @staticmethod
def remove_links(raw):
    return sub(r'https?://\S+', '', raw)

def remove_stopwords(raw):
    st = set(stopwords.words('english'))
    return " ".join([word for word in raw.split() if word not in st])


def remove_unnecessary_char(text): 
    text = remove_usernames(text)
    text = remove_punctuation(text)
    text = remove_links(text)
    text = remove_stopwords(text)
    return text

def get_clean_dataset(df):
    #  dir(nfx)
     df['text'] = df['text'].apply(nfx.remove_userhandles)
     df['text'] = df['text'].apply(nfx.remove_punctuations)
     df['text'] = df['text'].apply(nfx.remove_emojis)
     df['text'] = df['text'].apply(nfx.remove_hashtags)
     df['text'] = df['text'].apply(nfx.remove_html_tags)
     df['text'] = df['text'].apply(nfx.remove_stopwords)
     df['text'] = df['text'].apply(nfx.remove_urls)
     df['text'] = df['text'].apply(nfx.remove_phone_numbers)
     return df