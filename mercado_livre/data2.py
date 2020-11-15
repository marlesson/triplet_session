import luigi
import pandas as pd
import numpy as np
import os
import pickle
list_sum = sum

from mars_gym.data.task import BasePrepareDataFrames, BasePySparkTask
from mars_gym.data.utils import DownloadDataset
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
import random
from typing import Tuple, List, Union, Callable, Optional, Set, Dict, Any
from mars_gym.meta_config import *
from pyspark.ml.feature import StandardScaler

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_set, collect_list, lit, sum, udf, concat_ws, col, count, abs, date_format, \
    from_utc_timestamp, expr, min, max, mean, stddev
from pyspark.sql.functions import col, udf, size
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import explode, posexplode
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
import re
from unidecode import unidecode

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import  collect_list, lit, sum, udf, col, count, abs, max, lag, unix_timestamp
from pyspark.sql.functions import posexplode
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf, struct
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import CountVectorizer as CountVectorizerSpark
from pyspark.ml.linalg import Vectors
from pyspark.sql.window import Window
from pyspark.sql.functions import when
from pyspark.ml.feature import QuantileDiscretizer

from itertools import chain
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()
import dask.dataframe as dd

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre", "dataset")

BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "train_dataset_sample.jl")
BASE_TEST_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "test_dataset.jl")
BASE_METADATA_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "item_data.jl")

## AUX
# pad_history = F.udf(
#     lambda arr, size: list(reversed(([0] * (size - len(arr[:-1][:size])) + arr[:-1])))[:size], 
#     #lambda arr, size: list((arr[:size] + [0] * (size - len(arr[:size])))), 
#     ArrayType(IntegerType())
# )
def char_encode(text):
    vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    return [vocabulary.index(c)+1 if c in vocabulary else 0 for c in unidecode(str(text)).lower() ]

udf_char_encode = F.udf(char_encode, ArrayType(IntegerType()))

def pad_history(type):
    def _pad_history(arr, size, pad=0):
        return list(reversed(([pad] * (size - len(arr[:-1][:size])) + arr[:-1])))[:size]
    return udf(_pad_history, ArrayType(type))


def sample_items(item, item_list, size_available_list):
    return random.sample(item_list, size_available_list - 1) + [item]

def udf_sample_items(item_list, size_available_list):
    return udf(lambda item: sample_items(item, item_list, size_available_list), ArrayType(IntegerType()))


def concat(type):
    def concat_(*args):
        return list(set(chain.from_iterable((arg if arg else [] for arg in args))))
    return udf(concat_, ArrayType(type))

#####

from dateutil.parser import parse
from pyspark.sql.types import TimestampType
parse_date =  udf (lambda x: parse(x), TimestampType())


import itertools
import dask.bag as db
import json
import dask
from dask.distributed import Client, LocalCluster
# client=Client(
#         LocalCluster(
#             n_workers=1, 
#             threads_per_worker=12, 
#             memory_limit = '8GB', 
#             processes=False,  
#             resources={
#               'GPUS': 1,  # don't allow two tasks to run on the same worker
#             },))
#client=Client(processes=False)
#dask.config.set(scheduler='processes')

class PreProcessSessionDataset(BasePySparkTask):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_dataset.csv"))
    
    def get_path_dataset(self):
        return BASE_DATASET_FILE

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark    = SparkSession(sc)
        df = spark.read.json(self.get_path_dataset())#.sample(fraction=0.01, seed=42)

        if not 'item_bought' in df.columns:
            df = df.withColumn('item_bought', lit(-1))

        df = df.withColumn("session_id", F.monotonically_increasing_id())

        df = df.withColumn("event", explode(df.user_history))

        df = df.withColumn('event_info', col("event").getItem("event_info"))\
                .withColumn('event_timestamp', col("event").getItem("event_timestamp"))\
                .withColumn('event_type', col("event").getItem("event_type"))
        
        df = df.withColumn('event_timestamp', parse_date(col('event_timestamp')))

        df_view = df.select("session_id", "event_timestamp", "event_info", "event_type")

        df_buy  = df.groupBy("session_id").agg(max(df.event_timestamp).alias("event_timestamp"), 
                                            max(df.item_bought).alias("event_info"))
        df_buy  = df_buy.withColumn('event_type', lit("buy"))
        df_buy  = df_buy.withColumn('event_timestamp', df_buy.event_timestamp + F.expr('INTERVAL 30 seconds'))
        #df_buy  = df_buy.withColumn('event_timestamp', F.date_add(df_buy['event_timestamp'], 1))

        df = df_view.union(df_buy)

        df.orderBy(col('event_timestamp')).toPandas().to_csv(self.output().path, index=False)

class PreProcessSessionTestDataset(PreProcessSessionDataset):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset.csv"))
    
    def get_path_dataset(self):
        return BASE_TEST_DATASET_FILE

################################## Supervised ######################################

class TextDataProcess(luigi.Task):
    def requires(self):
        return PreProcessSessionDataset(), PreProcessSessionTestDataset()


    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_dataset__processed2.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset__processed2.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "item__processed2.csv"))

    def func_tokenizer(self, text):
        # print(text)

        text = str(text)

        # # Remove acentuação
        text = unidecode(text)

        # # lowercase
        text = text.lower()

        # #remove tags
        text = re.sub("<!--?.*?-->", "", text)

        # # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)
        text = re.sub('[^A-Za-z0-9]+', ' ', text)

        # # punk
        text = re.sub(r'[?|!|\'|#]', r'', text)
        text = re.sub(r'[.|,|:|)|(|\|/]', r' ', text)

        # Clean onde
        tokens = [t.strip() for t in text.split() if len(t) > 1]

        # remove stopwords
        #stopwords = self.load_stopwords()
        #tokens    = [t for t in tokens if t not in stopwords]

        if len(tokens) == 0:
            tokens.append("<pad>")
        # print(tokens)
        # print("")
        # if len(tokens) < 2:
        #    print(tokens)
        return tokens

    def add_more_information(self, df):
        print("add_more_information...")
        # add price

        # Split event_info
        def add_event_search(row):
          return row.event_info if row.event_type == "search" else " "

        df["event_search"] = df.apply(add_event_search, axis=1)

        def add_event_view(row):
          if row.event_type == "view":
            return int(row.event_info)
          elif row.event_type == "buy":
            return int(row.event_info)
          else:
            return -1

        df["event_view"] = df.apply(add_event_view, axis=1)

        return df

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        print("Load Data...")
        # Load
        df = dd.read_json(BASE_METADATA_FILE).repartition(npartitions=5)

        # Load train
        df_train = dd.read_csv(self.input()[0].path).repartition(npartitions=5).sample(frac=0.1)
        df_train = df_train.reset_index()#withColumn("idx", F.monotonically_increasing_id())
        df_train = self.add_more_information(df_train)

        # Load train
        df_test = dd.read_csv(self.input()[1].path).repartition(npartitions=5).sample(frac=0.1)
        df_test = df_test.reset_index()#withColumn("idx", F.monotonically_increasing_id())
        df_test = self.add_more_information(df_test)

        #Apply tokenizer 
        ## Metadada
        df["title"] = df.title.apply(char_encode, meta=('title', 'object'))

        ## Train
        df_train["event_search"] = df_train.event_search.apply(char_encode, meta=('event_search', 'object'))

        ## Test
        df_test["event_search"] = df_test.event_search.apply(char_encode, meta=('event_search', 'object'))

        df_train.visualize(filename='df_train_dask.svg')

        # Save
        df_train.compute().sort_values("event_timestamp").to_csv(self.output()[0].path, index=False)
        df_test.compute().sort_values("event_timestamp").to_csv(self.output()[1].path, index=False)
        df.compute().to_csv(self.output()[2].path, index=False)

class SessionPrepareDataset(luigi.Task):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    min_session_size: int = luigi.IntParameter(default=2)
    no_filter_data: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return TextDataProcess()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions),))


    def add_more_information(self, df):

        
            
        df['last_ItemID'] = df.groupby(by='SessionID')\
                              .apply(lambda df_g: df_g['ItemID'].shift(1), 
                                    meta=('last_ItemID', 'f8')).reset_index()['last_ItemID'] # .compute()

        df['last_ItemID'] = df.groupby(by='SessionID')['ItemID'].apply(lambda x: x.shift(-1))


        df.groupby(by='SessionID').assign(last_ItemID=df.ItemID.shift(1), 
                  last_event_search=df.event_search.shift(-1),
                  last_event_type=df.event_type.shift(-1))

        df.assign(last_ItemID=df.ItemID.shift(1), 
                  last_event_search=df.event_search.shift(-1),
                  last_event_type=df.event_type.shift(-1))

        df.last_ItemID.shi
        # Window Metrics
        w = Window.partitionBy('SessionID').orderBy('Timestamp')

        # Add Last Item
        
        df = df.withColumn("last_ItemID", lag(df.ItemID, 1).over(w).cast("int"))
        df = df.withColumn("last_event_search", lag(df.event_search, 1).over(w))
        df = df.withColumn("last_event_type", lag(df.event_type, 1).over(w))

        # Add step
        df = df.withColumn("step", lit(1))

        df = df.withColumn('step', F.sum('step').over(w))

        # Add Time diff and cum
        # https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-datetime-functions-b66de737950a
        df = df.withColumn("previous_t", lag(df.Timestamp, 1).over(w))\
                .withColumn("diff_Timestamp", F.floor((unix_timestamp(df.Timestamp) - unix_timestamp(col('previous_t')).cast("int"))/lit(60)) ) \
                .fillna(0, subset=['diff_Timestamp'])           
        
        # 1546308000 = '2019-01-01 00:00:00'
        df = df.withColumn('int_Timestamp', F.floor((df.Timestamp.cast("int")-lit(1546308000))/lit(60)) )
        df = df.withColumn('cum_Timestamp', F.sum('diff_Timestamp').over(w)).fillna(0, subset=['cum_Timestamp'])      

        # Add Scaler Price
        summary =  df.select([mean('price').alias('mu'), stddev('price').alias('sigma')])\
            .collect().pop()
        df = df.withColumn('price_norm', (df['price']-summary.mu)/summary.sigma)

        


        # Window Metrics Price
        w2 = Window.partitionBy('SessionID').orderBy('Timestamp').rowsBetween(Window.unboundedPreceding, -1)
        df = df.withColumn('min_price_norm', F.min('price_norm').over(w2))\
            .withColumn('max_price_norm', F.max('price_norm').over(w2))\
            .withColumn('mean_price_norm', F.mean('price_norm').over(w2))

        # add last search or view binary
        #...

        # Add default target
        df = df.withColumn('visit',lit(1))

        return df

    def add_history(self, df):
        int_pad_history = pad_history(IntegerType())
        str_pad_history = pad_history(StringType())
        float_pad_history = pad_history(FloatType())
        
        w = Window.partitionBy('SessionID').orderBy('Timestamp')#.rowsBetween(Window.unboundedPreceding, Window.currentRow)#.rangeBetween(Window.currentRow, 5)

        # History Step

        df = df.withColumn(
            'step_history', F.collect_list('step').over(w)
        )
        df = df.withColumn('step_history', int_pad_history(df.step_history, lit(self.history_window)))

        # History Time
        df = df.withColumn(
            'timestamp_history', F.collect_list('int_Timestamp').over(w)
        )
        df = df.withColumn('timestamp_history', int_pad_history(df.timestamp_history, lit(self.history_window)))

        # event_type
        df = df.withColumn(
            'event_type_history', F.collect_list('event_type').over(w)
        )
        df = df.withColumn('event_type_history', str_pad_history(df.event_type_history, lit(self.history_window)))

        # category_id
        df = df.withColumn(
            'category_id_history', F.collect_list('category_id').over(w)
        )
        df = df.withColumn('category_id_history', str_pad_history(df.category_id_history, lit(self.history_window)))

        # condition
        df = df.withColumn(
            'condition_id_history', F.collect_list('condition').over(w)
        )
        df = df.withColumn('condition_id_history', str_pad_history(df.condition_id_history, lit(self.history_window)))

        # domain_id
        df = df.withColumn(
            'domain_id_history', F.collect_list('domain_id').over(w)
        )
        df = df.withColumn('domain_id_history', str_pad_history(df.domain_id_history, lit(self.history_window)))

        # price
        df = df.withColumn(
            'price_history', F.collect_list('price_norm').over(w)
        )
        df = df.withColumn('price_history', float_pad_history(df.price_history, lit(self.history_window), lit(0.0)))

        # History Item
        df = df.withColumn(
            'ItemID_history', F.collect_list('ItemID').over(w)
        ).where(size(col("ItemID_history")) >= self.min_session_size)#\

        df = df.withColumn('ItemID_history', int_pad_history(df.ItemID_history, lit(self.history_window)))

        return df

    def filter(self, df):
        # filter date

        
        df['Timestamp']=dd.to_datetime(df.Timestamp,unit='ns')

        max_timestamp = df.Timestamp.max().compute()
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df[df.Timestamp >= init_timestamp]
        print(init_timestamp, max_timestamp)

        # Filter minin interactions
        df_item    = df.groupby("ItemID").count()
        df_item    = df_item[df_item.SessionID >= self.minimum_interactions].reset_index()
        #print("Filter minin interactions", df_item.count())

        # Filter session size
        df_session    = df.groupby("SessionID").count()
        df_session    = df_session[df_session.ItemID  >= self.min_session_size].reset_index()
        #print("Filter session size", df_session.count())
        df = df \
            .merge(df_item[["ItemID"]], on="ItemID") \
            .merge(df_session[["SessionID"]], on="SessionID")
        
        return df

    def add_available_items(self, df):
        all_items = list(df.select("ItemID").dropDuplicates().toPandas()["ItemID"])

        df = df.withColumn('AvailableItems', udf_sample_items(all_items, self.size_available_list)(col("ItemID")))

        return df

    def fillna(self, df):
        # fill
        df = df.fillna(value={'category_id': "-1", 
                        'condition': "-1", 
                        "domain_id": "-1", 
                        "product_id": "-1", 
                        "item_id": "-1",
                        "title": "[0]",
                        "price": np.mean})

        return df

    def input_path(self):
        return self.input()[0].path

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Item Metadada
        df_item  = dd.read_csv(self.input()[2].path)

        # Session
        df  = dd.read_csv(self.input_path())
        df = df.rename(columns={"session_id": "SessionID", "event_timestamp": "Timestamp",
                                "event_view": "ItemID"})

        if not self.no_filter_data:
            # Remove Search event
            #df = df.filter(df.event_type != "search")

            # Drop duplicate item in that same session
            df = df.drop_duplicates(subset=['SessionID', 'ItemID', 'event_type']) #TODO  ajuda ou atrapalha?
            
            # Filter 
            df = self.filter(df)
        
        # Join session with item metadada
        df = df.merge(df_item, left_on="ItemID", right_on="item_id", how="left")
        
        # Fillna
        df = self.fillna(df)

        # Add more information
        df = self.add_more_information(df)
        
        # add lag variable
        df = self.add_history(df)

        #df = self.add_available_items(df)

        # cast
        df = df.withColumn("last_ItemID", df.last_ItemID.cast("int"))

        if self.no_filter_data:
            df = df.filter(df.ItemID == -1)

        df.select("SessionID", "ItemID", "Timestamp", "event_type", "last_ItemID", "last_event_search",
            "last_event_type", "event_type_history", "ItemID_history", "timestamp_history",
            "category_id_history", "domain_id_history", "price_history")\
            .orderBy(col("SessionID")).toPandas().to_csv(self.output().path, index=False)

class SessionInteractionDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    days_test: int = luigi.IntParameter(default=1)
    index_mapping_path: str = luigi.Parameter(default=None)
    filter_only_buy: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def item_property(self) -> str:
        return "ItemID"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input().path

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.read_data_frame_path)#.sample(10000)
        
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        if self.filter_only_buy or data_key == 'TEST_GENERATOR': 
            df = df[df['event_type'] == 'buy']
        df['visit'] = 1
        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])
        ss
        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=self.days_test)

        return df[df[self.timestamp_property] < cutoff_date], df[df[self.timestamp_property] >= cutoff_date]

class SessionPrepareTestDataset(SessionPrepareDataset):
    sample_days: int = luigi.IntParameter(default=365)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=1)
    minimum_interactions: int = luigi.IntParameter(default=0)
    min_session_size: int = luigi.IntParameter(default=0)
    no_filter_data: bool = luigi.BoolParameter(default=True)

    def input_path(self):
        return self.input()[1].path

    def requires(self):
        return TextDataProcess()

#################################  Triplet ##############################

def serach_positive(uid, df, max_deep = 1, deep = 0, list_pos = []):
    if uid not in list_pos:
        list_pos.append(uid)
        #print(list_pos)
       
        if deep >= max_deep:
            return df.loc[uid].sub_a_b
        else:
            l = [serach_positive(i, df, max_deep, deep+1, list_pos) for i in df.loc[uid].sub_a_b]
            l.append(df.loc[uid].sub_a_b)
            return list_sum(l, [])
    else:
        return []

class CreateIntraSessionInteractionDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)
    pos_max_deep: int = luigi.IntParameter(default=1)

    # def requires(self):
    #     return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_intra_session_train_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "item_positive_interaction_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "item_id_index_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "session_index_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
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
        
        # Filter more then 1 ocurrence for positive interactions
        df = df.filter(col("total_ocr_dupla") >= 1)
    
        df = df\
            .groupby("ItemID_A")\
            .agg(F.collect_set("ItemID_B").alias("sub_a_b"))

        # df_b = df\
        #     .groupby("ItemID_B")\
        #     .agg(F.collect_set("ItemID_A").alias("sub_b"))

        # df = df.join(df_a, "ItemID_A").join(df_b, "ItemID_B").cache()

        # concat_int_arrays = concat(IntegerType())
        # df = df.withColumn("sub_a_b", concat_int_arrays("sub_a", "sub_b"))#.show(truncate=False)
        # return df
        df = df.withColumnRenamed("ItemID_A", "ItemID")
        #df = df.withColumn("ItemID_COPY",df.ItemID)

        df = df.toPandas().set_index('ItemID')
        print(df)

        sub_pos = []
        for i, row in df.iterrows():
            l = serach_positive(row.name, df, max_deep = self.pos_max_deep, deep=0, list_pos=[])
            sub_pos.append(list(np.unique(l)))
        
        df['sub_a_b_all'] = sub_pos

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        #parans
        min_itens_per_session  = 2
        max_itens_per_session  = self.max_itens_per_session
        min_itens_interactions = self.min_itens_interactions # Tupla interactions
        max_relative_pos       = self.max_relative_pos

        spark    = SparkSession(sc)
        df = spark.read.option("delimiter", ";").csv(BASE_DATASET_FILE, header=True, inferSchema=True)
        df = df.withColumnRenamed("sessionId", "SessionID")\
            .withColumnRenamed("eventdate", "Timestamp")\
            .withColumnRenamed("itemId", "ItemID")\
            .withColumn("Timestamp", (col("Timestamp").cast("long") + col("timeframe").cast("long")/1000).cast("timestamp"))\
            .orderBy(col('Timestamp'), col('SessionID'), col('timeframe')).select("SessionID", "ItemID", "Timestamp", "timeframe")
                   
        # Drop duplicate item in that same session
        df       = df.dropDuplicates(['SessionID', 'ItemID'])

        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()

        df       = df.groupby("SessionID").agg(
                    max("Timestamp").alias("Timestamp"),
                    collect_list("ItemID").alias("ItemIDs"),
                    count("ItemID").alias("total"))

        # Filter Interactions
        df = df.filter(df.total >= min_itens_per_session).cache()

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
        df = df.select('SessionID', 'Timestamp', 'ItemID_A', 'pos_A', 
                        'ItemID_B', 'pos_B', 'relative_pos')\
                .distinct()\
                .filter(df.ItemID_A != df.ItemID_B).cache()

        # # Filter duplicates
        # udf_join = F.udf(lambda s,x,y : "_".join(sorted([str(s), str(x),str(y)])) , StringType())
        # df = df.withColumn('key', udf_join('SessionID', 'ItemID_A','ItemID_B'))
        # df = df.dropDuplicates(["key"])

        # Calculate and filter probs ocorrence
        df_probs = self.get_df_tuple_probs(df)
        df = df.join(df_probs, (df.ItemID_A == df_probs._ItemID_A) & (df.ItemID_B == df_probs._ItemID_B))

        # Add positive interactoes
        df_positive = self.add_positive_interactions(df)

        # Filter confidence
        df = df.filter(col("total_ocr_dupla") >= min_itens_interactions)\
               .filter(col("relative_pos") <= max_relative_pos)\
               .filter(col("pos_A") <= self.max_itens_per_session)
                   

        # df = df.select("SessionID", 'Timestamp', 'ItemID_A', 'pos_A',
        #                 'ItemID_B', 'pos_B', 'relative_pos', 
        #                 'total_ocr', 'total_ocr_dupla', 'prob', 'sub_a_b')\
        #         .dropDuplicates(['ItemID_A', 'ItemID_B', 'relative_pos']) # TODO is it right?
        df = df.select("SessionID", 'Timestamp', 'ItemID_A', 'ItemID_B', 'relative_pos', 
                        'total_ocr', 'total_ocr_dupla')\
                .dropDuplicates(['ItemID_A', 'ItemID_B', 'relative_pos']) # TODO is it right?

        df.select("ItemID_A").dropDuplicates().toPandas().to_csv(self.output()[2].path, index_label="item_idx")
        df.select("SessionID").dropDuplicates().toPandas().to_csv(self.output()[3].path, index_label="session_idx")
        df.write.parquet(self.output()[0].path)
        df_positive.to_csv(self.output()[1].path)

class IntraSessionInteractionsDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)
    days_test: int = luigi.IntParameter(default=1)
    pos_max_deep: int = luigi.IntParameter(default=1)    
    filter_first_interaction: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return CreateIntraSessionInteractionDataset(
                        max_itens_per_session=self.max_itens_per_session,
                        sample_days=self.sample_days,
                        min_itens_interactions=self.min_itens_interactions,
                        max_relative_pos=self.max_relative_pos,
                        pos_max_deep=self.pos_max_deep)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_parquet(self.read_data_frame_path)#.sample(10000)

        # TODO
        if self.filter_first_interaction:
            df = df.groupby(['ItemID_A', 'ItemID_B']).head(1).reset_index(drop=True)
        
        #df["ItemID"]        = df.ItemID_A
        #df['sub_a_b']        = df['sub_a_b'].apply(list)
        df['available_arms'] = None
        df["visit"]          = 1
        
        df_session           = df[['SessionID']].drop_duplicates().reset_index().rename(columns={"index":'SessionIDX'})
        
        df = df.merge(df_session).drop(['SessionID'], axis=1)
        df = df.rename(columns={"ItemID_A":'ItemID'})

        return df

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return self.input()[1].path
        
    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        print(data_key)
        print(df.describe())
        
        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])

        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=self.days_test)

        return df[df[self.timestamp_property] < cutoff_date], df[df[self.timestamp_property] >= cutoff_date]

################################## Interactions ######################################

