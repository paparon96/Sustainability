# Data collection/processing steps:
1. Run `time python gdelt_data_extract.py` (around 1-2 minutes for 1 year's data)
2. Run `time python url_data_extract.py` (around 1-2 minutes for 1 year's data)
3. Run `time python nlp_preprocess.py` (less than 1 minute for 1 year's data)
4. Run `time python keyword_matrix_generation.py` (less than 1 minute for 1 year's data)


# Running the app
* Execute `streamlit run dashboard.py`
* Navigate to `http://localhost:8501` in your browser
