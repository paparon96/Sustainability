FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app/sustainability/Carbon_price_prediction/App

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY ./data/research_paper/new_merged_dataset.csv /app/sustainability/Carbon_price_prediction/Data/new_merged_dataset.csv
COPY ./data/research_paper/signals /app/sustainability/Carbon_price_prediction/Data/signals

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
