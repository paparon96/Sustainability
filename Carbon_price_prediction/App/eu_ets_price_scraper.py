from bs4 import BeautifulSoup
import requests
import urllib
import pandas as pd


url = 'https://sandbag.be/index.php/carbon-price-viewer/'
r = requests.get(url)
print(r.status_code)
# print(r.text)

tmp = urllib.request.urlopen(url).read()
# print(tmp)
soup = BeautifulSoup(tmp)
price = soup.find_all("p", {"class": "price-value-today"})
print(price)


df = pd.read_csv("https://raw.githubusercontent.com/ember-climate/ember-data-api/main/data/api_day_ahead_price.csv")
print(df.head())
