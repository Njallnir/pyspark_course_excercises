import numpy as np
import pyspark
import pyspark.pandas as ps

from pyspark.sql.functions import *
from pyspark.sql.types import *


@pandas_udf(DoubleType(),PandasUDFType.SCALAR)
def calculate_video_score(views, likes, dislikes,comment_likes_count):
    return\
        likes.fillna(0)\
        + 0.42 * views.fillna(0) * 0.2\
        - 2 * dislikes.fillna(0)\
        + comment_likes_count.fillna(0) * 0.01


@pandas_udf(DoubleType(), PandasUDFType.GROUPED_AGG)
def pandas_mean(x):
    return x.mean()


def split_udf(value, separator):
    @udf(returnType=ArrayType(StringType()))
    def split_udf_inner(string):
        return string.split(separator)
    return split_udf_inner(value)
