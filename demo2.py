import os
os.environ["PYSPARK_PYTHON"] = "python3"

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

movie_rating = spark.read.load("tables/ratings.csv", format='csv', header = True)


header = movie_rating.take(1)[0]
print(header)
rating_data_df = movie_rating.select(movie_rating['userId'], movie_rating['movieId'], movie_rating['rating'])

#rating_data = [tuple(x) for x in rating_data_df.values]
print(rating_data_df.take(3))

train, validation, test = rating_data_df.randomSplit([6,2,2],seed = 7856)
train.cache()
validation.cache()
test.cache()
print(train.take(3))

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_param:
            # write your approach to train ALS model
            # make prediction
            # get the rating result
            # get the RMSE

            als = ALS(maxIter=num_iters, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating",
                      coldStartStrategy="drop")
            model = als.fit(train_data)
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print("Root-mean-square error = " + str(rmse))

            error = None
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, error))
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))
    return best_model

