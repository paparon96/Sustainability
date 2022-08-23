import aiohttp
import asyncio

import pandas as pd

async def fetch(session, url):
    async with session.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        return await response.text()

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
        print(pd.DataFrame({'url': urls, 'html': htmls}))

if __name__ == '__main__':
    asyncio.run(main())
