import scrapy
class ReutersScrape(scrapy.Spider):
    name = 'reutersScrape'
    start_urls = ['https://www.reuters.com/news/archive/businessnews?view=page&page=1&pageSize=10']
    def parse(self, response):
        for story in response.css('.news-headline-list'):
            yield {'title':story.css('.story-title').get() }