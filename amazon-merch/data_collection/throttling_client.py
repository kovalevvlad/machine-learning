import datetime

import logging
import requests
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import numpy as np


class ThrottlingChromeLikeClient:
    def __init__(self, mean_request_period_seconds):
        self.mean_request_period_seconds = mean_request_period_seconds
        self.next_allowed_request_time = datetime.datetime.now()
        self.session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=1,
                        status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def __getattr__(self, method):
        def invoke_throttled_requests(*args, **kwargs):
            kwargs["headers"] = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, sdch, br",
                "Accept-Language": "en-GB,en-US;q=0.8,en;q=0.6",
                "Cache-Control": "max-age=0",
                "Connection": "keep-alive",
                "Host": "www.amazon.com",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36"
            }

            now = datetime.datetime.now()
            if now < self.next_allowed_request_time:
                seconds_to_sleep = (self.next_allowed_request_time - now).seconds + 1
                logging.debug("Sleeping for {} seconds until {}".format(seconds_to_sleep, self.next_allowed_request_time))
                time.sleep(seconds_to_sleep)

            self.next_allowed_request_time = datetime.datetime.now() + self._next_wait_timedelta()
            logging.debug("Next request can be sent at {}".format(self.next_allowed_request_time))
            return getattr(self.session, method)(*args, **kwargs)

        return invoke_throttled_requests

    def _next_wait_timedelta(self):
        seconds = np.random.exponential(self.mean_request_period_seconds)
        return datetime.timedelta(seconds=int(seconds) + 1)


def _retry_n_times(f, times, sleep_seconds):
    retried = 0
    while retried < times:
        try:
            return f()
        except Exception as e:
            logging.error("Got {} when issuing a request. Will retry.".format(e))
            time.sleep(sleep_seconds)

    raise RuntimeError("Reached the maximum retry limit.")
