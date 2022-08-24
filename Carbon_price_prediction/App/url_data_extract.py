import aiohttp
import asyncio

import pandas as pd

from goose3 import Goose

extractor = Goose()

async def fetch(session, url):
    async with session.get(url) as response:
        if response.status != 200:
            return None
        return await response.text()

async def fetch_all(session, urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(session, url))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

async def main():
    filtered_events = pd.read_csv('./data/gdelt_events.csv')
    urls = list(filtered_events.sourceurl)
    async with aiohttp.ClientSession() as session:
        htmls = await fetch_all(session, urls)
        df = pd.DataFrame({'url': urls, 'html': htmls})
        df['text'] = df['html'].apply(lambda x: extractor.extract(raw_html=x).cleaned_text if x is not None else x)
        df = filtered_events.merge(df[['url', 'text']],
                                   left_on='sourceurl', right_on='url')
        df[['sqldate', 'text']].to_csv('./data/article_text.csv', index=False)

if __name__ == '__main__':
    asyncio.run(main())
