import pytest

from chispa import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from video_analytics.functions import *


@pytest.fixture(scope='session')
def spark():
    return SparkSession.builder \
        .master("local") \
        .appName("chispa") \
        .getOrCreate()


def test_calculate_video_score(spark):
    data = [
        (100, 5, 1, 6, 11.46),
        (47, 9, 0, 0, 12.948),
        (0, 0, 0, 0, 0.),
        (10000, 5, 1222, 13, -1598.87),
        (42, 42, 42, 42, -38.052)
    ]
    df = spark.createDataFrame(data, ['views', 'likes', 'dislikes', 'comment_likes_count', 'expected_score']) \
        .withColumn(
        'video_score',
        calculate_video_score(
            col('views'),
            col('likes'),
            col('dislikes'),
            col('comment_likes_count')
        )
    )
    assert_column_equality(df, 'video_score', 'expected_score')


def test_pandas_mean(spark):
    data = [
        (13.2, 12),
        (13.2, 4),
        (13.2, 5),
        (13.2, 44),
        (13.2, 1)
    ]
    df = spark.createDataFrame(data, ['expected_mean', 'x']) \
        .groupBy(col('expected_mean')) \
        .agg(pandas_mean(col('x')).alias('mean'))
    assert_column_equality(df, 'mean', 'expected_mean')


def test_split_udf(spark):
    data = [
        ('aaa|bbb|ccc', ['aaa', 'bbb', 'ccc']),
        ('a|b|c|1|', ['a', 'b', 'c', '1', ''])
    ]
    df = spark.createDataFrame(data, ['string', 'expected_result']) \
        .withColumn('result', split_udf(col('string'), '|'))
    assert_column_equality(df, 'result', 'expected_result')
