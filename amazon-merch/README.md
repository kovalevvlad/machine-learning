## Summary
This project attempts to improve a seller's performance on Amazon Merch
platform by scraping listing data information from Amazon and then
performing basic data analysis on the available data.

## Amazon Merch

> **Merch by Amazon** - Sell your designs on the world's
largest marketplace with no upfront investment or costs.

Here is the official service page - https://merch.amazon.com/landing

## Scraper

The scraper lives at [data_collection/amazon_crawler.py](data_collection/amazon_crawler.py).
 It takes a start page and the page number (e.g. 1 for the first page).
 From that point the scraper loads every single listing found by the search
 ignoring non-merch products. It stores rank, price, title, description
 and other information about each listing into a SQLlite database until
 it is manually stopped. It deals with various issues such as 503 errors
 and Amazon captchas by retrying (in the case of captchas a massive
 back-off solves the problem).

## Analysis

Simple analysis is perfomed by [analysis.py](analysis.py). It considers
effects of varying the price and title on the product rank.


