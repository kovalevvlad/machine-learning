import datetime
import re

import logging
import urlparse
import time
from lxml import etree
from listing import Listing
from sql_data_source import SqlLiteDataSource
from throttling_client import ThrottlingChromeLikeClient


class AmazonCrawler:
    def __init__(self, source, current_page_url, current_page_number=1):
        self.next_page_url = current_page_url
        self.source = source
        self.next_page_number = current_page_number + 1
        # anti-throttling measure
        self.no_requests_before = datetime.datetime.now()
        self.remaining_links_on_current_page = []
        self.throttling_client = ThrottlingChromeLikeClient(10)

    def crawl(self):
        while True:
            listing = self._next_listing()
            logging.debug(listing.__dict__)
            self.source.add_listing(listing)

    def _next_listing(self):
        while True:
            if len(self.remaining_links_on_current_page) == 0:
                self._go_to_next_page()

            next_listing_url = self.remaining_links_on_current_page.pop()
            next_listing = self._listing_for_url(next_listing_url)
            if next_listing is not None:
                # If next_listing is None, it means that it's not a novelty T-shirt
                return next_listing

    def _go_to_next_page(self):
        logging.debug("Navigating to page {} ({})".format(self.next_page_number, self.next_page_url))
        retry_attempt = 0
        while True:
            response = self.throttling_client.get(self.next_page_url)
            response.raise_for_status()
            content_tree = etree.HTML(response.content)
            self.remaining_links_on_current_page = content_tree.xpath('//div[@class="s-item-container"]/div[1]//a/@href')
            next_page_url_list = content_tree.xpath('//div[@id="bottomBar"]//a[text()="{}"]/@href'.format(self.next_page_number))
            if len(next_page_url_list) > 0:
                self.next_page_url = "https://www.amazon.com" + next_page_url_list[0]
                self.next_page_number += 1
                return
            else:
                # Amazon issues a Captcha when you send too many requests. This cools down the rate at which we send
                # requests.
                sleep_duration = (2 ** retry_attempt) * 60
                logging.info("Retry #{}. Sleeping for {} seconds.".format(retry_attempt + 1, sleep_duration))
                time.sleep(sleep_duration)
                retry_attempt += 1

    def _listing_for_url(self, url):
        response = self.throttling_client.get(url)
        response.raise_for_status()
        content_tree = etree.HTML(response.content)

        is_novelty_t = any(
            "Lightweight, Classic fit, Double-needle sleeve and bottom hem" in point
            for point in content_tree.xpath('//div[@id="feature-bullets"]//li//span/text()'))

        if not is_novelty_t:
            return None

        brand_search_url = urlparse.urlparse(content_tree.xpath('//a[@id="brand"]/@href')[0])
        brand_search_url_dict = dict(t.split("=") for t in brand_search_url.query.split("&"))
        brand_name_key = "field-lbr_brands_browse-bin"
        is_big_brand = brand_name_key in brand_search_url_dict

        if is_big_brand:
            brand = brand_search_url_dict[brand_name_key]
            logging.debug("Skipping {} since it might be a big brand and may have intellectual property that skews the results".format(brand))
            return None

        def clean_first_xpath(tree, xpath):
            tree_elements = tree.xpath(xpath)
            if len(tree_elements) == 0:
                return None
            else:
                return tree_elements[0].strip()

        title = clean_first_xpath(content_tree, '//span[@id="productTitle"]/text()')

        price_str = clean_first_xpath(content_tree, '//div[@id="price"]//span/text()')
        price_regex = r"^\$([0-9]+\.[0-9]{2}).*"
        assert re.match(price_regex, price_str), price_str
        price = float(re.match(price_regex, price_str).groups()[0])

        description = clean_first_xpath(content_tree, '//div[@id="productDescription"]/p/text()')

        review_count_str = clean_first_xpath(content_tree, '//span[@id="acrCustomerReviewText"]/text()')
        if review_count_str is not None:
            assert re.match("^[0-9]+ customer reviews?$", review_count_str), review_count_str
            review_count = int(review_count_str.split(" ")[0])

            review_score_str = clean_first_xpath(content_tree, '//div[@id="averageCustomerReviews"]//a//span[@class="a-icon-alt"]/text()')
            assert re.match(r"^[0-5](\.[0-9]+)? out of 5 stars$", review_score_str), review_score_str
            review_score = float(review_score_str.split(" ")[0])
        else:
            review_count = 0
            review_score = float("nan")

        rank_str = clean_first_xpath(content_tree, '//li[@id="SalesRank"]/text()[contains(.,"#")]')
        rank_regex = ".*#([0-9,]+) in Clothing, Shoes [&] Jewelry.*"
        if rank_str is None:
            rank = None
        else:
            assert re.match(rank_regex, rank_str), rank_str
            rank = int(re.match(rank_regex, rank_str).groups()[0].replace(",", ""))

        asin_spans = content_tree.xpath('//div[@id="detailBullets"]//span')
        asin_span_mask = [s.text is not None and "ASIN" in s.text for s in asin_spans]
        asin_label_span_index = asin_span_mask.index(True)
        asin = asin_spans[asin_label_span_index + 1].text
        assert len(asin) == 10, asin

        return Listing(title, price, description, review_count, review_score, rank, asin, url)


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

with SqlLiteDataSource("data.sqlt") as sql_source:
    crawler = AmazonCrawler(sql_source, "https://www.amazon.com/s/ref=sr_pg_1?fst=p90x%3A1%2Cas%3Aoff&rh=k%3ALightweight%5Cc+Classic+fit%5Cc+Double-needle+sleeve+and+bottom+hem%2Cn%3A7141123011%2Cp_6%3AATVPDKIKX0DER&bbn=7141123011&sort=price-desc-rank&keywords=Lightweight%2C+Classic+fit%2C+Double-needle+sleeve+and+bottom+hem&ie=UTF8&qid=1500029674", 40)
    crawler.crawl()
