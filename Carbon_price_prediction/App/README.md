# Create environment for the app
Run the below commands in your terminal (while the current directory is the `/App` folder)
1. `conda env create -f environment.yml`
2. `conda activate carbon_dashboard`
3. `python -c "import nltk; nltk.download('all')"`

# Data collection/processing steps:
The below steps can be run compactly through this shell script: `time ./data_update.sh` (around 2 minutes for 1 day's data).

1. Run `time python gdelt_data_extract.py` (around 30 seconds for 1 day's data)
2. Run `time python url_data_extract.py` (around 30 seconds for 1 day's data)
3. Run `time python nlp_preprocess.py` (around 15 seconds for 1 day's data)
  * **Note that the existing `lemmatized_merged_articles.csv` file needs to be downloaded from Google Drive (more specifically from the `git_lfs_replacement` folder on Google Drive, because it was too large for Git LFS) and copied to the `/data` folder first before running this script! And then after running the script, the updated `lemmatized_merged_articles.csv` file needs to be uploaded to Google Drive (more specifically to the `git_lfs_replacement` folder on Google Drive)!**
4. Run `time python keyword_matrix_generation.py` (around 4 minutes for the full dataset)
5. Run `time python tf_idf_generation.py` (around a few seconds for 1 day's data)


# Running the app
* Execute `streamlit run Carbon_price_prediction/App/dashboard.py` (from the root of the repository)
* Navigate to `http://localhost:8501` in your browser
