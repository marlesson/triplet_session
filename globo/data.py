import luigi
import pandas as pd
import numpy as np
import os
from mars_gym.data.task import BasePrepareDataFrames, BasePySparkTask
from mars_gym.data.utils import DownloadDataset
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
import random
from typing import Tuple, List, Union, Callable, Optional, Set, Dict, Any
from mars_gym.meta_config import *

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_set, collect_list, lit, sum, udf, concat_ws, col, count, abs, date_format, \
    from_utc_timestamp, expr, min, max
from pyspark.sql.functions import col, udf, size
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import explode, posexplode
from itertools import chain
from datetime import datetime, timedelta

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "globo")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "globo", "dataset")

BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "globo", "archive", 'clicks', 'clicks', '*.csv')

## AUX
pad_history = F.udf(
    lambda arr, size: [0] * (size - len(arr[:-1][:size])) + arr[:-1][:size], 
    ArrayType(IntegerType())
)

def sample_items(item, item_list, size_available_list):
    return random.sample(item_list, size_available_list - 1) + [item]

def udf_sample_items(item_list, size_available_list):
    return udf(lambda item: sample_items(item, item_list, size_available_list), ArrayType(IntegerType()))


def concat(type):
    def concat_(*args):
        return list(set(chain.from_iterable((arg if arg else [] for arg in args))))
    return udf(concat_, ArrayType(type))

#####


################################## Supervised ######################################

class SessionPrepareDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions),))

    def add_history(self, df):
        
        w = Window.partitionBy('SessionID').orderBy('Timestamp')#.rangeBetween(Window.currentRow, 5)

        df = df.withColumn(
            'ItemIDHistory', F.collect_list('ItemID').over(w)
        ).where(size(col("ItemIDHistory")) >= 2)#\

        df = df.withColumn('ItemIDHistory', pad_history(df.ItemIDHistory, lit(self.history_window)))

        return df

    def filter(self, df):
        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()

        # Filter minin interactions
        df_item    = df.groupBy("ItemID").count()
        df_item    = df_item.filter(col('count') >= self.minimum_interactions)

        # Filter session size
        df_session    = df.groupBy("SessionID").count()
        df_session    = df_session.filter(col('count') >= 2)

        df = df \
            .join(df_item, "ItemID", how="inner") \
            .join(df_session, "SessionID", how="inner")

        return df

    def add_available_items(self, df):
        all_items = list(df.select("ItemID").dropDuplicates().toPandas()["ItemID"])

        df = df.withColumn('AvailableItems', udf_sample_items(all_items, self.size_available_list)(col("ItemID")))

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        #  ["SessionID", "Timestamp", "ItemID", "Category"]
        #  DataFrame[_c0: int, _c1: timestamp, _c2: int, _c3: string]                      


        spark    = SparkSession(sc)
        df = spark.read.csv(BASE_DATASET_FILE, header=True, inferSchema=True)
        df = df.withColumnRenamed("session_id", "SessionID")\
            .withColumnRenamed("click_timestamp", "Timestamp_")\
            .withColumnRenamed("click_article_id", "ItemID")\
            .withColumn("Timestamp",F.from_unixtime(col("Timestamp_")/lit(1000)).cast("timestamp"))\
            .orderBy(col('Timestamp')).select("SessionID", "ItemID", "Timestamp", "Timestamp_").filter(col('Timestamp') < '2017-10-16 24:59:59')
                    
        # Drop duplicate item in that same session
        df = df.dropDuplicates(['SessionID', 'ItemID'])

        df = self.filter(df)
        df = self.add_history(df)
        df = self.add_available_items(df)

        df = df.withColumn('visit',lit(1))

        df.toPandas().to_csv(self.output().path, index=False)

class SessionInteractionDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    days_test: int = luigi.IntParameter(default=1)

    def requires(self):
        return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input().path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])

        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=self.days_test)

        return df[df[self.timestamp_property] <= cutoff_date.date()], df[df[self.timestamp_property] > cutoff_date.date()]


#################################  Triplet ##############################


class CreateIntraSessionInteractionDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)
    # def requires(self):
    #     return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_intra_session_train_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos)))

    def get_df_tuple_probs(self, df):

        df_tuple_count  = df.groupby("ItemID_A","ItemID_B").count()
        df_count        = df.groupby("ItemID_A").count()\
                            .withColumnRenamed("count", "total")\
                            .withColumnRenamed("ItemID_A", "_ItemID_A")

        df_join         = df_tuple_count.join(df_count, df_tuple_count.ItemID_A == df_count._ItemID_A).cache()
        df_join         = df_join.withColumn("prob", col("count")/col("total"))

        df_join  = df_join.select("ItemID_A", 'ItemID_B', 'count', 'total', 'prob')\
                    .withColumnRenamed("ItemID_A", "_ItemID_A")\
                    .withColumnRenamed("ItemID_B", "_ItemID_B")\
                    .withColumnRenamed("count", "total_ocr_dupla")\
                    .withColumnRenamed("total", "total_ocr").cache()

        return df_join

    def add_positive_interactions(self, df):

        df_a = df\
            .groupby("ItemID_A")\
            .agg(F.collect_set("ItemID_B").alias("sub_a"))

        df_b = df\
            .groupby("ItemID_B")\
            .agg(F.collect_set("ItemID_A").alias("sub_b"))

        df = df.join(df_a, "ItemID_A").join(df_b, "ItemID_B").cache()

        concat_int_arrays = concat(IntegerType())
        df = df.withColumn("sub_a_b", concat_int_arrays("sub_a", "sub_b"))#.show(truncate=False)
        
        return df
        
    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        #parans
        min_itens_per_session  = 2
        max_itens_per_session  = self.max_itens_per_session
        min_itens_interactions = self.min_itens_interactions # Tupla interactions
        max_relative_pos       = self.max_relative_pos

        spark    = SparkSession(sc)
        df = spark.read.csv(BASE_DATASET_FILE, header=True, inferSchema=True)
        df = df.withColumnRenamed("session_id", "SessionID")\
            .withColumnRenamed("click_timestamp", "Timestamp_")\
            .withColumnRenamed("click_article_id", "ItemID")\
            .withColumn("Timestamp",F.from_unixtime(col("Timestamp_")/lit(1000)).cast("timestamp"))\
            .orderBy(col('Timestamp')).select("SessionID", "ItemID", "Timestamp", "Timestamp_").filter(col('Timestamp') < '2017-10-16 24:59:59')
               

        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()

        # Drop duplicate item in that same session
        df       = df.dropDuplicates(['SessionID', 'ItemID'])

        df       = df.groupby("SessionID").agg(
                    max("Timestamp").alias("Timestamp"),
                    collect_list("ItemID").alias("ItemIDs"),
                    count("ItemID").alias("total"))


        # Filter Interactions
        df = df.filter(df.total >= min_itens_per_session)\
                .filter(df.total <=  max_itens_per_session).cache()

        # Filter position in list
        df_pos = df.select(col('SessionID').alias('_SessionID'),
                                    posexplode(df.ItemIDs))

        # Explode A
        df = df.withColumn("ItemID_A", explode(df.ItemIDs))
        df = df.join(df_pos,
                    (df.SessionID == df_pos._SessionID) & (df.ItemID_A == df_pos.col))\
                .select('SessionID', 'Timestamp', 'ItemID_A', 'pos', 'ItemIDs')\
                .withColumnRenamed('pos', 'pos_A')

        # Explode B
        df = df.withColumn("ItemID_B", explode(df.ItemIDs))
        df = df.join(df_pos,
                    (df.SessionID == df_pos._SessionID) & (df.ItemID_B == df_pos.col))\
                .withColumnRenamed('pos', 'pos_B')

        df = df.withColumn("relative_pos", abs(df.pos_A - df.pos_B))


        # Filter  distincts
        df = df.select('SessionID', 'Timestamp', 'ItemID_A', 'pos_A', 'ItemID_B', 'pos_B', 'relative_pos')\
                .distinct()\
                .filter(df.ItemID_A != df.ItemID_B).cache()

        # Filter duplicates
        #udf_join = F.udf(lambda s,x,y : "_".join(sorted([str(s), str(x),str(y)])) , StringType())
        #df = df.withColumn('key', udf_join('SessionID', 'ItemID_A','ItemID_B'))
        #df = df.dropDuplicates(["key"])

        # Calculate and filter probs ocorrence
        df_probs = self.get_df_tuple_probs(df)

        # Add positive interactoes
        df = self.add_positive_interactions(df)

        df = df.join(df_probs, (df.ItemID_A == df_probs._ItemID_A) & (df.ItemID_B == df_probs._ItemID_B))
        df = df.filter(col("total_ocr_dupla") >= min_itens_interactions)\
               .filter(col("relative_pos") <= max_relative_pos)

        df = df.select("SessionID", 'Timestamp', 'ItemID_A', 'pos_A',
                        'ItemID_B', 'pos_B', 'relative_pos', 
                        'total_ocr', 'prob', 'sub_a_b')\
                .dropDuplicates(['ItemID_A', 'ItemID_B'])

        df.write.parquet(self.output().path)

class IntraSessionInteractionsDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)

    def requires(self):
        return CreateIntraSessionInteractionDataset(
                        max_itens_per_session=self.max_itens_per_session,
                        sample_days=self.sample_days,
                        min_itens_interactions=self.min_itens_interactions,
                        max_relative_pos=self.max_relative_pos)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_parquet(self.read_data_frame_path)#.sample(200000)
        df["ItemID"]        = df.ItemID_A

        return df

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input().path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        df["visit"]         = 1.0
        df['sub_a_b']       = df['sub_a_b'].apply(list)

        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])

        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        days=1    
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=days)

        df[df[self.timestamp_property] <= cutoff_date.date()]

        #size = len(df)
        #cut = int(size - size * test_size)

        return df[df[self.timestamp_property] <= cutoff_date.date()], df[df[self.timestamp_property] > cutoff_date.date()]

################################## Interactions ######################################

