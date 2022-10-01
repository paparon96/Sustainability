import pandas as pd

from utils import tf_idf


# Parameters / Constants
ngrams = [1, 2, 3]
grouping = True

# Data import
if grouping:
    group_mapping = pd.read_csv('./data/keyword_lists/group_mapping.csv',
                                index_col=0).squeeze()


dfs = [pd.read_csv(f'./data//merged_articles_carbon_keyword_term_document_matrix_ngram_{ngram}.csv',
                   index_col=0)
       for ngram in ngrams]
df = pd.concat(dfs, axis=1)
print(df.shape)
print(df.head())

# Data preprocesing

# Sanity check
print(min(df.index))
print(max(df.index))

# Potential grouping
if grouping:
    df = df.rename(columns=group_mapping)
    df = df.groupby(by=df.columns, axis=1).apply(lambda g: g.sum(axis=1))

# Aggregate by dates
agg_df = df.groupby(df.index).sum()

# Generate TF-IDF scores
tf_idf_df = tf_idf(agg_df)

# Export results
grouping_flag = '_grouped' if grouping else ''
tf_idf_df.to_csv(f'./data/tf_idf_gdelt_lemmatized{grouping_flag}_custom_keywords.csv')

# Aggregated keyword matrix
agg_keyword_index_df = df.groupby(df.index).sum().sum(axis=1)
agg_keyword_index_tf_idf_df = tf_idf(agg_keyword_index_df)
agg_keyword_index_tf_idf_df.to_csv(f'./data/tf_idf_gdelt_lemmatized_aggregated_keywords.csv')
