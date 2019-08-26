import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import math

import os
os.environ["PYSPARK_PYTHON"] = "python3"

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

movies = spark.read.load("tables/movies.csv", format='csv', header = True)
ratings = spark.read.load("tables/ratings.csv", format='csv', header = True)
links = spark.read.load("tables/links.csv", format='csv', header = True)
tags = spark.read.load("tables/tags.csv", format='csv', header = True)

ratings.show(5)

# data agg view for rating
rating_user_agg = ratings.groupBy("userID").count().toPandas()['count'] # easier to implement using series
rating_movie_agg = ratings.groupBy("movieId").count().toPandas()['count']
tmp1 = rating_user_agg.min()
tmp2 = rating_movie_agg.min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

tmp1 = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

print('\nThere are {} users give out rating'.format(rating_user_agg.count()))
print('\nThere are {} Movies in the database'.format(movies.count()))

print('There are {} movies been rated'.format(rating_movie_agg.count()))
rated_id_list = ratings.groupBy("movieId").count().select("movieId").toPandas().movieId.tolist()
# ~ isin list
df_movie_notRated = movies.filter(~movies.movieId.isin(rated_id_list))
print('\nThere are {} movies not been rated with which the name will be displayed below'.format(df_movie_notRated.count()))
print(df_movie_notRated.select("title").toPandas().title)

print("----------------List Movie genres-----------------\n")
print(movies.groupBy("genres").count().select("genres").toPandas())

from pyspark.ml.recommendation import ALS