import aiohttp
import asyncio
import requests

import pandas as pd

from goose3 import Goose

extractor = Goose()

async def fetch(session, url):
    async with session.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        # article = extractor.extract(raw_html=response.text())
        # return await article.cleaned_text
        return await response.text()
        # return await response.content

async def fetch_all(session, urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(session, url))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

async def main():
    urls = ['https://www.eubusiness.com/topics/commission/top-stories/w22-30',
            'https://www.washingtonexaminer.com/policy/defense-national-security/us-denies-sanctions-food-crisis-russia-africa-west']
    async with aiohttp.ClientSession() as session:
        htmls = await fetch_all(session, urls)
        df = pd.DataFrame({'url': urls, 'html': htmls})
        df['text'] = df['html'].apply(lambda x: extractor.extract(raw_html=x).cleaned_text)
        df.to_csv('./data/article_text.csv', index=False)

if __name__ == '__main__':
    asyncio.run(main())

    # Old approach output
    response = requests.get('https://www.eubusiness.com/topics/commission/top-stories/w22-30', timeout=5)
    article = extractor.extract(raw_html=response.content)
    text = article.cleaned_text

    # New approach output
    df = pd.read_csv('./data/article_text.csv')
    text2 = str(df['text'].iloc[0])

    assert text == text2
