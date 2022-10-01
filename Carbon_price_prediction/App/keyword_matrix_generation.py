import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Parameters / Constants
ngrams = range(1, 4)

# Import data
text_df = pd.read_csv( "./data/lemmatized_merged_articles.csv", index_col=0)
print(text_df.head())

# Term-document matrix generation
for ngram in ngrams:
    print(f"{ngram}-grams")

    # Count Vectorizer
    vect = CountVectorizer(ngram_range=(ngram, ngram))
    vects = vect.fit_transform(text_df.lemmatized_text)

    td_matrix_cols = vect.get_feature_names()
    sparse_matrix = vects

    # Import climate change keyword list
    carbon_keywords = pd.read_csv("./data/keyword_lists/revised_keyword_list.csv")
    carbon_keywords['keywords'] = carbon_keywords['keywords'].\
                                  apply(lambda x: x.lower())

    # Replace logic for lemmatization
    carbon_keywords['keywords'] = carbon_keywords['keywords'].\
    replace({"emissions trading system": "emission trading system",
             "emissions trading scheme": "emission trading scheme"})

    carbon_keywords_index = {carbon_keyword: td_matrix_cols.index(carbon_keyword)
                         for carbon_keyword in carbon_keywords.squeeze().values
                         if carbon_keyword in td_matrix_cols}

    carbon_keyword_matrix = sparse_matrix.tocsr()[:,
    list(carbon_keywords_index.values())].todense()

    carbon_keyword_df = pd.DataFrame(carbon_keyword_matrix,
    columns=list(carbon_keywords_index.keys()), index=text_df['date'].values)
    print(carbon_keyword_df.head())

    print((carbon_keyword_df > 0).sum(axis=0))
    carbon_keyword_df.to_csv(f'./data/merged_articles_carbon_keyword_term_document_matrix_ngram_{ngram}.csv')
