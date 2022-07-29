import iso639.exceptions
import scrapy
import os
from iso639 import Lang


class WikiLangSpider(scrapy.Spider):
    """
    Webcrawler (scrapy.Spider) for extracting languages, language codes (iso 639-3) and their family trees from Wikipedia
    """
    name = "wikilangspider"
    bibles_path = "data/pbc"
    start_urls = list(sorted(set(["https://en.wikipedia.org/wiki/ISO_639:{}".format(file[:3]) for file in os.listdir(bibles_path)])))

    def parse(self, response):
        iso = response.request.url[-3:]

        language_tree = response.xpath("//table[@class='infobox vevent']//a[@title='Language family']/ancestor::th/following-sibling::td/descendant::*/text()").getall()
        language_tree = [lang.strip() for lang in language_tree if len(lang.strip()) > 0]

        try:
            name = Lang(iso).name
        except iso639.exceptions.DeprecatedLanguageValue:
            name = ""

        yield {"name": name,
               "iso639-3": response.request.url[-3:],
               "family-tree": language_tree}
