import unicodedata
import string

import pandas as pd
import numpy as np


def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")


def tf_idf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates TF-IDF scores based on term-document matrix

    Parameters:
        df (pandas DataFrame):term-document dataframe

    Returns:
        tf_idf_df (pandas DataFrame): Dataframe of TF-IDF scores

    """

    # Count term occurences across documents (dates in ur case)
    nt = (df > 0).sum(axis=0)

    # Broadcast to dataframe for compatible shapes
    nt = (df < 0) + nt

    # Get number of documents
    N = len(df)

    # Implementation based on the 2. recommended option here: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    tf_idf_df = np.log(1 + df) * np.log(1 + N / nt)

    return tf_idf_df
