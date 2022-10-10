# Create environment for the app
Run the below commands in your terminal (while the current directory is the `/App` folder)
1. `conda env create -f environment.yml`
2. `conda activate carbon_dashboard`
3. `python3 -c "import nltk; nltk.download('all')"`

# Data collection/processing steps:
The below steps can be run compactly through this shell script: `time ./data_update.sh` (around 2 minutes for 1 day's data).

1. Run `time python gdelt_data_extract.py` (around 30 seconds for 1 day's data)
2. Run `time python url_data_extract.py` (around 30 seconds for 1 day's data)
3. Run `time python nlp_preprocess.py` (around 15 seconds for 1 day's data)
4. Run `time python keyword_matrix_generation.py` (around 5 seconds for 1 day's data)
5. Run `time python tf_idf_generation.py` (around a few seconds for 1 day's data)


# Running the app
* Execute `streamlit run dashboard.py`
* Navigate to `http://localhost:8501` in your browser
