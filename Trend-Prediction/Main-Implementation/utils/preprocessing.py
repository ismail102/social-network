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
    #  df['text'] = df['text'].apply(nfx.remove_stopwords)
     df['text'] = df['text'].apply(nfx.remove_urls)
     df['text'] = df['text'].apply(nfx.remove_phone_numbers)
     df['text'] = df['text'].apply(nfx.remove_numbers)
     df['text'] = df['text'].apply(nfx.remove_accents)
     df['text'] = df['text'].apply(nfx.remove_bad_quotes)
     df['text'] = df['text'].apply(nfx.remove_multiple_spaces)
     df['text'] = df['text'].apply(nfx.remove_terms_in_bracket)
     df['text'] = df['text'].apply(nfx.remove_special_characters)
     df['text'] = df['text'].apply(nfx.remove_non_ascii)
     df['text'] = df['text'].apply(nfx.remove_shortwords)
     df['text'] = df['text'].apply(nfx.remove_puncts)
     df['text'] = df['text'].apply(nfx.remove_currency_symbols)
     df['text'] = df['text'].apply(nfx.remove_md5sha)
     df['text'] = df['text'].apply(nfx.remove_street_address)
     df['text'] = df['text'].apply(nfx.remove_dates)
    #  df['text'] = df['text'].apply(nfx.fix_contractions)
     return df

# Define a dictionary of common contractions and their expanded forms
contraction_dict = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "I'd've": "I would have",
    "she'd've": "she would have",
    "should've": "should have",
    "could've": "could have",
    "might've": "might have",
    "would've": "would have"
    # Add more contractions and their expansions as needed
}

# Function to expand contractions using the dictionary
def expand_contractions(text):
    for contraction, expansion in contraction_dict.items():
        text = text.replace(contraction, expansion)
    return text

if __name__ == '__main__':
     print("#Preprocession...")
