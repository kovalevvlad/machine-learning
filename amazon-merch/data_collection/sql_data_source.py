import os
import sqlite3


class SqlLiteDataSource:
    def __init__(self, data_file):
        self.data_file = data_file
        self.connection = None
        if not os.path.exists(data_file):
            # Init file and DB
            open(data_file, "a").close()
            with sqlite3.connect(data_file) as con:
                cursor = con.cursor()
                cursor.execute("""
                CREATE TABLE listing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    price REAL NOT NULL,
                    description TEXT NULL,
                    review_count INTEGER NOT NULL,
                    review_score REAL NULL,
                    rank INTEGER NOT NULL,
                    asin TEXT NOT NULL,
                    url TEXT NOT NULL,
                    CONSTRAINT asin_unique UNIQUE (asin));""")
                cursor.execute("CREATE UNIQUE INDEX asin_index ON listing(asin);")
                con.commit()

    def __enter__(self):
        self.connection = sqlite3.connect(self.data_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def add_listing(self, listing):
        cursor = self.connection.cursor()
        same_listing_count = cursor.execute("SELECT COUNT(1) FROM listing WHERE asin = :asin", {"asin": listing.asin}).fetchall()[0][0]
        if same_listing_count == 0:
            cursor.execute("""INSERT INTO listing (title, price, description, review_count, review_score, rank, asin, url)
                                       values (:title, :price, :description, :review_count, :review_score, :rank, :asin, :url)""",
                           {
                               "title": listing.title,
                               "price": listing.price,
                               "description": listing.description,
                               "review_count": listing.review_count,
                               "review_score": listing.review_score,
                               "rank": listing.rank,
                               "asin": listing.asin,
                               "url": listing.url
                           })
            self.connection.commit()

