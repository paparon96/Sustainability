import pandas as pd

import os.path
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from utils import remove_accents


# Import data
raw_text = pd.read_csv( "./data/article_text.csv")
raw_text = raw_text.rename(columns={'sqldate': 'date'})
print(raw_text.head())
print(raw_text.shape)

# Parameters / Constants
# POS (Parts Of Speech) for: nouns, adjectives, verbs and adverbs
DI_POS_TYPES = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
POS_TYPES = list(DI_POS_TYPES.keys())

# Constraints on tokens
MIN_STR_LEN = 3
RE_VALID = '[a-zA-Z]'

# NLP preprocessing

# Convert DataFrame columns to list of tuples
raw_text_iter = list(zip(raw_text.globaleventid, raw_text.date, raw_text.text))

# Get stopwords, stemmer and lemmatizer
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Process all article texts
lemmatized_results = []

counter = 0

for globaleventid, date, text in raw_text_iter:

    if counter % 1000 == 0:
        print(f"Iteration: {counter + 1}/{len(raw_text_iter)}")

    if not isinstance(text, str):
        continue
    # Tokenize by sentence, then by lowercase word
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    lemmas = []
    # Process all tokens per article text
    for token in tokens:
        # Remove accents
        t = remove_accents(token)

        # Remove punctuation
        t = str(t).translate(string.punctuation)

        # Add token that represents "no lemmatization match"
        lemmas.append("-") # this token will be removed if a lemmatization match is found below

        # Process each token
        if t not in stopwords:
            if re.search(RE_VALID, t):
                if len(t) >= MIN_STR_LEN:
                    # Note that the POS (Part Of Speech) is necessary as input to the lemmatizer
                    # (otherwise it assumes the word is a noun)
                    pos = nltk.pos_tag([t])[0][1][:2]
                    pos2 = 'n'  # set default to noun
                    if pos in DI_POS_TYPES:
                      pos2 = DI_POS_TYPES[pos]

                    stem = stemmer.stem(t)
                    lem = lemmatizer.lemmatize(t, pos=pos2)  # lemmatize with the correct POS

                    if pos in POS_TYPES:
                        # Remove the "-" token and append the lemmatization match
                        lemmas = lemmas[:-1]
                        lemmas.append(lem)

    # Build list of strings from lemmatized tokens
    str_lemmas = ' '.join(lemmas)
    lemmatized_results.append((globaleventid, date, str_lemmas))

    # Increment counter
    counter += 1


lemmatized_text_df = pd.DataFrame(lemmatized_results)
lemmatized_text_df.columns = ['globaleventid', 'date', 'lemmatized_text']

print(lemmatized_text_df.shape)
print(lemmatized_text_df.head())


# Export results
fname = './data/lemmatized_merged_articles.csv'
if os.path.isfile(fname):
    print("Appending to existing file")
    text_df = pd.read_csv( "./data/lemmatized_merged_articles.csv", index_col=0)
    print(text_df.shape)
    print(lemmatized_text_df.shape)
    lemmatized_text_df = pd.concat([text_df, lemmatized_text_df])
    print(lemmatized_text_df.shape)
    lemmatized_text_df.to_csv(fname)

else:
    print("Creating new file")
    lemmatized_text_df.to_csv(fname)
