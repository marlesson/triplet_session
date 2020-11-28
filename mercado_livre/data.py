import luigi
import pandas as pd
import numpy as np
import os
import pickle
list_sum = sum
from pyspark import SparkConf

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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gensim

from itertools import chain
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()
from scipy import stats
import pickle

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre", "dataset")

BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "train_dataset.jl")
BASE_TEST_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "test_dataset.jl")
BASE_METADATA_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "item_data.jl")

WORD_MODEL  = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(BASE_DIR, "assets", 'mercadolivre-100d.bin'), binary=True)
WORD_MODEL.add(['UNK', 'PAD'], [np.random.random(100), np.random.random(100)])
WORD_UNK  = 18751
WORD_PAD  = 18752
WORD_DICT = 18753

DATA_COLUMNS =  ["SessionID",
                "ItemID",
                "domain_idx",
                "Timestamp",
                "event_type_idx",
                "last_category_idx",
                "last_product_id",
                "last_ItemID",
                "last_ItemID_title",
                "last_event_search",
                "last_event_type_idx",
                "last_domain_idx",
                "step",
                #"int_Timestamp",
                "cum_Timestamp",
                "last_price_norm",
                "diff_price_norm",
                "min_last_price_norm",
                "max_last_price_norm",
                "mean_last_price_norm",
                "sum_last_price_norm",
                #"step_history",
                "timestamp_history",
                "cum_timestamp_history",
                #"event_type_idx_history",
                "category_idx_history",
                #"condition_idx_history",
                "domain_idx_history",
                "product_id_history",
                "price_history",
                "ItemID_history",
                "title_search_history",
                "domain_count",
                "item_id_count",
                "mode_category_idx_history",
                "mode_condition_idx_history",
                "mode_domain_idx_history",
                "mode_product_id_history",
                "mode_event_type_idx_history",
                "count_mode_category_idx_history",
                "count_mode_condition_idx_history",
                "count_mode_domain_idx_history",
                "count_mode_product_id_history",
                "count_mode_event_type_idx_history",
                "count_event_type_idx_history__search",
                "count_event_type_idx_history__view",
                "count_condition_idx__new",
                "count_condition_idx__used",
                "perc_newlest_search",
                "perc_event_view"]

DEBUG       = False

ML_BUY      = 1 if DEBUG else 2
ML_SEARCH   = 2 if DEBUG else 3
ML_VIEW     = 3 if DEBUG else 4

if DEBUG:
    BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "sample_train_dataset.jl")
    BASE_TEST_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "sample_test_dataset.jl")

import pandas as pd

def deDupeDfCols(df, separator=''):
    newcols = []

    for col in df.columns:
        if col not in newcols:
            newcols.append(col)
        else:
            for i in range(2, 1000):
                if (col + separator + str(i)) not in newcols:
                    newcols.append(col + separator + str(i))
                    break

    return df.toDF(*newcols)

def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]

def toPandas(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
    repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand

def word_tokenizer(text):
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
        tokens.append("PAD") 
    
    return [WORD_MODEL.vocab[word].index if word in WORD_MODEL.wv else WORD_MODEL.vocab["UNK"].index for word in tokens]

udf_word_tokenizer = F.udf(word_tokenizer, ArrayType(IntegerType()))


def char_encode(text):
    vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    return [vocabulary.index(c)+1 if c in vocabulary else 0 for c in unidecode(text).lower() ]
udf_char_encode = F.udf(char_encode, ArrayType(IntegerType()))


def pad_history(type):
    def _pad_history(arr, size, pad=-1):
        if isinstance(arr, str):
            arr = eval(arr)

        return list(reversed(([pad] * (size - len(arr[:-1][:size])) + arr[:-1])))[:size]
    return udf(_pad_history, ArrayType(type))

def pad_history_norm(type):
    def _pad_history(arr, size, pad=-1):
        if isinstance(arr, str):
            arr = eval(arr)

        return list((arr + [pad] * (size - len(arr[:size]))))[:size]
    return udf(_pad_history, ArrayType(type))


def sample_items(item, item_list, size_available_list):
    return random.sample(item_list, size_available_list - 1) + [item]

def udf_sample_items(item_list, size_available_list):
    return udf(lambda item: sample_items(item, item_list, size_available_list), ArrayType(IntegerType()))

def concat(type):
    def concat_(*args):
        return list(set(chain.from_iterable((arg if arg else [] for arg in args))))
    return udf(concat_, ArrayType(type))

def stat_float_hist(func):
    def _stat_float_hist(arr, pad=0.0):
        arr = np.array(arr)
        arr = arr[arr != pad]
        if len(arr) == 0:
            return 0.0

        return float(func(arr))
    return udf(_stat_float_hist, FloatType())

def moda_hist(arr, pad='-1'):
    arr = np.array(arr)
    arr = arr[arr != pad]
    if len(arr) == 0:
        return ""

    return str(stats.mode(arr)[0][0])
udf_moda_hist = udf(moda_hist, StringType())
         
def moda_count_hist(arr, pad='-1'):
    arr = np.array(arr)
    arr = arr[arr != pad]
    if len(arr) == 0:
        return 0

    return int(stats.mode(arr)[1][0])
udf_moda_count_hist = udf(moda_count_hist, IntegerType())
    
def count_hist(arr, val='0'):
    arr = np.array(arr)
    arr = arr[arr == val]
    if len(arr) == 0:
        return 0
    
    return int(stats.mode(arr)[1][0])
udf_count_hist = udf(count_hist, IntegerType())    


#####

from dateutil.parser import parse
from pyspark.sql.types import TimestampType
parse_date =  udf (lambda x: parse(x), TimestampType())

class PreProcessSessionDataset(BasePySparkTask):
    test_size: int = luigi.IntParameter(default=100 if DEBUG else 1000)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_train_dataset.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "local_session_test_dataset_{}.csv".format(self.test_size)))

    def get_path_dataset(self):
        return BASE_DATASET_FILE

    def explode(self, df):
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
        
        return df.orderBy(col('event_timestamp'))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark    = SparkSession(sc)
        df = spark.read.json(self.get_path_dataset())#.sample(fraction=0.01, seed=42)

        if not 'item_bought' in df.columns:
            df = df.withColumn('item_bought', lit(0))

        df = df.withColumn("session_id", F.monotonically_increasing_id()).cache()

        if self.test_size > 0:
            df_train = df.limit(df.count() - self.test_size)
            df_test  = df.limit(self.test_size)
        else: 
            df_train = df
            df_test  = df.limit(1)

        df_train = self.explode(df_train)
        df_test  = self.explode(df_test)

        #if DEBUG: # 68719488980
        #    df = df.filter(df.session_id.isin(["42949685985", "68719488980", "94489310665", "77309420244"]))

        toPandas(df_train).to_csv(self.output()[0].path, index=False)
        toPandas(df_test).to_csv(self.output()[1].path, index=False)

class PreProcessSessionTestDataset(PreProcessSessionDataset):
    test_size: int = luigi.IntParameter(default=0)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "local_session_test_dataset_{}.csv".format(self.test_size)))

    
    def get_path_dataset(self):
        return BASE_TEST_DATASET_FILE

################################## Supervised ######################################

class TextDataProcess(BasePySparkTask):
    def setup(self, conf: SparkConf):
        conf.set("spark.local.dir", os.path.join("output", "spark"))
        conf.set("spark.driver.maxResultSize", self._get_available_memory())
        conf.set("spark.sql.shuffle.partitions", os.cpu_count())
        conf.set("spark.default.parallelism", os.cpu_count())
        conf.set("spark.executor.memory", self._get_available_memory())
        #conf.set("spark.sql.execution.arrow.enabled","true")
        pass


    def requires(self):
        return PreProcessSessionDataset(), PreProcessSessionTestDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_train_dataset__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "local_session_test_dataset__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "item__processed.csv"))

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

        # Split event_info
        df = df.withColumn("event_search",
            when(df.event_type == "search", col("event_info")).
            otherwise(""))

        df = df.withColumn("event_view",
            when(df.event_type == "view", col("event_info")).
            when(df.event_type == "buy", col("event_info")).            
            otherwise(-1))

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)
        print("Load Data...")

        # Load
        spark    = SparkSession(sc)
        df = spark.read.json(BASE_METADATA_FILE)
        #df_text = df.select(["item_id", "title"]).toPandas()
        df = df.withColumn("price", col("price").cast("float"))

        # Load train
        df_train = spark.read.option("delimiter", ",").csv(self.input()[0][0].path, header=True, inferSchema=True)#.sample(True, 0.1)
        df_train = df_train.withColumn("idx", F.row_number().over(Window.partitionBy().orderBy(df_train['event_timestamp'])))
        df_train = self.add_more_information(df_train)

        # Load test
        df_test = spark.read.option("delimiter", ",").csv(self.input()[1][0].path, header=True, inferSchema=True)#.sample(True, 0.1)
        df_test = df_test.withColumn("idx", F.row_number().over(Window.partitionBy().orderBy(df_test['event_timestamp'])))
        df_test = self.add_more_information(df_test)
        
        # Load local_test
        df_test2 = spark.read.option("delimiter", ",").csv(self.input()[0][1].path, header=True, inferSchema=True)#.sample(True, 0.1)
        df_test2 = df_test2.withColumn("idx", F.row_number().over(Window.partitionBy().orderBy(df_test2['event_timestamp'])))
        df_test2 = self.add_more_information(df_test2)
                
        #Apply tokenizer 
        word_pad_history = pad_history_norm(IntegerType())

        ## Metadada
        text_column = "title"
        df = df.withColumn(text_column, udf_word_tokenizer(col(text_column)))
        pad_history
        ## Train
        text_column = "event_search"
        df_train = df_train.withColumn(text_column, udf_word_tokenizer(col(text_column)))        

        ## Test
        text_column = "event_search"
        df_test = df_test.withColumn(text_column, udf_word_tokenizer(col(text_column)))                

        ## Test 2
        text_column = "event_search"
        df_test2 = df_test2.withColumn(text_column, udf_word_tokenizer(col(text_column)))                



        # Reindex
        df_category_id  = df.select("category_id").orderBy(col('category_id'))\
                                .dropDuplicates().cache()
        df_category_id  = df_category_id.withColumn("category_idx", F.row_number().over(Window.partitionBy().orderBy(df_category_id['category_id'])))
        df_condition    = df.select("condition").orderBy(col('condition'))\
                                .dropDuplicates().cache()
        df_condition    = df_condition.withColumn("condition_idx", F.row_number().over(Window.partitionBy().orderBy(df_condition['condition'])))
        df_domain_id    = df.select("domain_id").orderBy(col('domain_id'))\
                                .dropDuplicates().cache()
        df_domain_id    = df_domain_id.withColumn("domain_idx", F.row_number().over(Window.partitionBy().orderBy(df_domain_id['domain_id'])))


        df_item = df\
                .join(df_category_id, ["category_id"], how="inner")\
                .join(df_condition, ["condition"], how="inner")\
                .join(df_domain_id, ["domain_id"], how="inner")

        df_event_type   = df_train.select("event_type").orderBy(col('event_type'))\
                            .dropDuplicates().cache()
        df_event_type   = df_event_type.withColumn("event_type_idx", F.row_number().over(Window.partitionBy().orderBy(df_event_type['event_type'])))

        df_train = df_train\
            .join(df_event_type, ["event_type"], how="inner")          
        df_test = df_test\
            .join(df_event_type, ["event_type"], how="inner")      
        df_test2 = df_test2\
            .join(df_event_type, ["event_type"], how="inner")      


        # Add Domain COunt
        df_train    = df_train.join(df_item.select("item_id", "domain_idx"), 
                        df_train.event_view == df_item.item_id, how="left")
        
        df_item_count   = df_train.select("item_id").groupBy("item_id").count()\
                                .withColumnRenamed("count", "item_id_count")\
        
        df_domain_count = df_train.select("domain_idx").groupBy("domain_idx").count()\
                                .withColumnRenamed("count", "domain_count")\
        


        df_test     = df_test.withColumn("domain_count", lit(0))
        df_test     = df_test.join(df_item_count, df_test.event_view == df_item_count.item_id, how='left')

        df_test2    = df_test2.withColumn("domain_count", lit(0))
        df_test2    = df_test2.join(df_item_count, df_test2.event_view == df_item_count.item_id, how='left')

        df_train    = df_train.join(df_domain_count, "domain_idx", how='left')
        df_train    = df_train.join(df_item_count, "item_id", how='left')

        
        #print(df_train.show(2))
        #return 

        print("Save!!!")
        columns = ["event_type","session_id","event_timestamp","event_search","event_view","event_type_idx","domain_count", "item_id_count"]
        df_test = df_test.select(columns).orderBy(col('event_timestamp'))
        df_test2 = df_test2.select(columns).orderBy(col('event_timestamp'))
        df_train = df_train.select(columns).orderBy(col('event_timestamp'))

        # Save
        df_category_id.toPandas().to_csv(DATASET_DIR+"/index_category_id.csv", index=False)
        df_condition.toPandas().to_csv(DATASET_DIR+"/index_condition.csv", index=False)
        df_domain_id.toPandas().to_csv(DATASET_DIR+"/index_domain_id.csv", index=False)
        df_event_type.toPandas().to_csv(DATASET_DIR+"/index_event_type.csv", index=False)

        toPandas(df_test).to_csv(self.output()[1].path, index=False)
        toPandas(df_test2).to_csv(self.output()[2].path, index=False)
        toPandas(df_train).to_csv(self.output()[0].path, index=False)
        toPandas(df_item).to_csv(self.output()[3].path, index=False)


        return

class SessionPrepareDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    min_session_size: int = luigi.IntParameter(default=2)
    no_filter_data: bool = luigi.BoolParameter(default=False)

    def setup(self, conf: SparkConf):
        conf.set("spark.local.dir", os.path.join("output", "spark"))
        conf.set("spark.driver.maxResultSize", self._get_available_memory())
        conf.set("spark.sql.shuffle.partitions", os.cpu_count())
        conf.set("spark.default.parallelism", os.cpu_count())
        conf.set("spark.executor.memory", self._get_available_memory())

    def requires(self):
        return TextDataProcess()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}_min_s={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions, self.min_session_size),)),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}_min_s={}.parquet"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions, self.min_session_size),))                    


    def add_more_information(self, df):
        # Window Metrics
        w  = Window.partitionBy('SessionID').orderBy('Timestamp')

        w2 = Window.partitionBy(['SessionID', 'event_type_click']).orderBy('Timestamp')

        # Add Last Item
        
        df = df.withColumn("last_ItemID", lag(df.ItemID, 1).over(w2).cast("int")).fillna(-1, subset=['last_ItemID'])
        df = df.withColumn("last_category_idx", lag(df.category_idx, 1).over(w2).cast("int")).fillna(-1, subset=['last_category_idx'])
        df = df.withColumn("last_product_id", lag(df.product_id, 1).over(w2).cast("int")).fillna(-1, subset=['last_product_id'])

        word_pad_history = pad_history_norm(IntegerType())
        df = df.withColumn("last_ItemID_title", lag(df.title, 1).over(w2)).fillna("[]", subset=['last_ItemID_title'])
        df = df.withColumn("last_ItemID_title", word_pad_history(col("last_ItemID_title"), lit(15), lit(WORD_PAD)))

        #w3 = Window.partitionBy(['SessionID', 'event_type_click']).orderBy(col('event_type_click').asc(), 'Timestamp')
        df = df.withColumn("last_event_search",   lag(df.event_search, 1).over(w) ).fillna("[]", subset=['last_event_search'])
        df = df.withColumn("last_event_search",   word_pad_history(col("last_event_search"), lit(15), lit(WORD_PAD)))

        df = df.withColumn("last_event_type_idx",   lag(df.event_type_idx, 1).over(w))

        df = df.withColumn("last_title_search",  when(col("last_event_type_idx") == lit(ML_SEARCH), col("last_event_search")).otherwise(col("last_ItemID_title")))
        df = df.withColumn("last_title_search",  word_pad_history(col("last_title_search"), lit(15), lit(WORD_PAD)))

        # Add step
        df = df.withColumn("step", lit(1))
        df = df.withColumn('step', F.sum('step').over(w))
        df = df.filter(df.step > 1)
        
        # Add Time diff and cum
        # https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-datetime-functions-b66de737950a
        df = df.withColumn("previous_t", lag(df.Timestamp, 1).over(w))\
                .withColumn("diff_Timestamp", F.floor((unix_timestamp(df.Timestamp) - unix_timestamp(col('previous_t')).cast("int"))/lit(60)) ) \
                .fillna(0, subset=['diff_Timestamp'])           
        
        # 1546308000 = '2019-01-01 00:00:00'
        df = df.withColumn('int_Timestamp', F.floor((df.Timestamp.cast("int")-lit(1546308000))/lit(60)) )
        df = df.withColumn('cum_Timestamp', F.sum('diff_Timestamp').over(w)).fillna(0, subset=['cum_Timestamp'])      
        #df = df.withColumn('cum_Timestamp', df.cum_Timestamp/F.max('cum_Timestamp'))      

        # Add Scaler Price
        summary =  df.filter(df.event_type_idx == ML_VIEW).select([mean('price').alias('mu'), stddev('price').alias('sigma')])\
                        .collect().pop()
        df = df.withColumn('price_norm', F.round((df['price']-summary.mu)/summary.sigma, 4))
        #df = df.withColumn('price_norm', F.round(df['price'], 4))
        df = df.withColumn('last_price_norm', F.round(lag(df.price_norm, 1).over(w2), 4)).fillna(0, subset=['last_price_norm'])      
        df = df.withColumn("diff_price_norm",  F.round(lag(col('price_norm'), 2).over(w2)-df.last_price_norm, 4))

        # Last Domain
        df = df.withColumn("last_domain_idx", lag(df.domain_idx, 1).over(w2).cast("int")).fillna(-1, subset=['last_domain_idx'])
        
        # Window Metrics Price
        #w2 = Window.partitionBy(['SessionID', 'event_type_click']).orderBy('Timestamp').rowsBetween(Window.unboundedPreceding, -1)
        df = df.withColumn('min_last_price_norm', F.round(F.min('last_price_norm').over(w2), 4))\
                .withColumn('max_last_price_norm', F.round(F.max('last_price_norm').over(w2), 4))\
                .withColumn('mean_last_price_norm', F.round(F.mean('last_price_norm').over(w2), 4))\
                .withColumn('sum_last_price_norm', F.round(F.sum('last_price_norm').over(w2), 4))\
                .fillna(0.0, subset=["diff_price_norm" ,"min_last_price_norm" ,"max_last_price_norm",
                                    "mean_last_price_norm" ,"sum_last_price_norm"])      
        
        # add last search or view binary
        #...

        # Add default target
        df = df.withColumn('visit',lit(1))

        return df#.cache()

    def add_history(self, df):
        int_pad_history = pad_history(IntegerType())
        str_pad_history = pad_history(StringType())
        float_pad_history = pad_history(FloatType())
        
        w = Window.partitionBy(['SessionID']).orderBy('Timestamp')#.rowsBetween(Window.unboundedPreceding, Window.currentRow)#.rangeBetween(Window.currentRow, 5)
        w2  = Window.partitionBy(['SessionID', 'event_type_click']).orderBy('Timestamp')#.rowsBetween(Window.unboundedPreceding, Window.currentRow)#.rangeBetween(Window.currentRow, 5)
        
        # History Step

        df = df.withColumn(
            'step_history', F.collect_list('step').over(w2)
        )
        df = df.withColumn('step_history', int_pad_history(df.step_history, lit(self.history_window)))

        # History Time
        df = df.withColumn(
            'timestamp_history', F.collect_list('int_Timestamp').over(w2)
        )
        df = df.withColumn('timestamp_history', int_pad_history(df.timestamp_history, lit(self.history_window)))

        # History Time 2
        df = df.withColumn(
            'cum_timestamp_history', F.collect_list('cum_Timestamp').over(w2)
        )
        df = df.withColumn('cum_timestamp_history', int_pad_history(df.cum_timestamp_history, lit(self.history_window)))

        # event_type_idx
        df = df.withColumn(
            'event_type_idx_history', F.collect_list('event_type_idx').over(w)
        )
        df = df.withColumn('event_type_idx_history', str_pad_history(df.event_type_idx_history, lit(self.history_window)))

        # category_idx
        df = df.withColumn(
            'category_idx_history', F.collect_list('category_idx').over(w2)
        )
        df = df.withColumn('category_idx_history', str_pad_history(df.category_idx_history, lit(self.history_window)))

        # condition
        df = df.withColumn(
            'condition_idx_history', F.collect_list('condition_idx').over(w2)
        )
        df = df.withColumn('condition_idx_history', str_pad_history(df.condition_idx_history, lit(self.history_window)))

        # domain_id
        df = df.withColumn(
            'domain_idx_history', F.collect_list('domain_idx').over(w2)
        )
        df = df.withColumn('domain_idx_history', str_pad_history(df.domain_idx_history, lit(self.history_window)))

        # product_id
        df = df.withColumn(
            'product_id_history', F.collect_list('product_id').over(w2)
        )
        df = df.withColumn('product_id_history', str_pad_history(df.product_id_history, lit(self.history_window)))

        # price
        df = df.withColumn(
            'price_history', F.collect_list(F.round(col('price_norm'), 4)).over(w2)
        )
        df = df.withColumn('price_history', float_pad_history(df.price_history, lit(self.history_window), lit(0.0)))

        # History Item
        df = df.withColumn(
            'ItemID_history', F.collect_list('ItemID').over(w2)
        )#\

        #History Item
        w3 = Window.partitionBy(['SessionID']).orderBy('Timestamp').rowsBetween(-5, Window.currentRow-1)#.rangeBetween(Window.currentRow, 5) -3
        df = df.withColumn(
            'title_search_history', F.collect_list('last_title_search').over(w3)
        )#\
        

        df = df.withColumn('ItemID_history', int_pad_history(df.ItemID_history, lit(self.history_window)))

        return df

    def add_more_information_after_history(self, df):

        # Moda Features
        for c in ['category_idx_history', 'condition_idx_history', "domain_idx_history", "product_id_history", "event_type_idx_history"]:
            df = df.withColumn('mode_'+c,  udf_moda_hist(col(c))).fillna("0", subset=['mode_'+c])      

        # Count Mode
        for c in ['category_idx_history', 'condition_idx_history', "domain_idx_history", "product_id_history", "event_type_idx_history"]:
            df = df.withColumn('count_mode_'+c,  udf_moda_count_hist(col(c))).fillna(0, subset=['count_mode_'+c])      

        # Count Value Mode
        # 1 - buy
        # 2 - search
        # 3 - view
        #         
        df = df.withColumn('count_event_type_idx_history__search',  udf_count_hist(col("event_type_idx_history"), lit(str(ML_SEARCH))))
        df = df.withColumn('count_event_type_idx_history__view',  udf_count_hist(col("event_type_idx_history"), lit(str(ML_VIEW))))
        df = df.withColumn('count_condition_idx__new',  udf_count_hist(col("condition_idx_history"), lit('2')))
        df = df.withColumn('count_condition_idx__used', udf_count_hist(col("condition_idx_history"), lit('4')))
        df = df.withColumn('perc_newlest_search',   F.round(when(col('count_condition_idx__new') > 0, (col('count_condition_idx__new'))/(col('count_condition_idx__new')+col('count_condition_idx__used'))).otherwise(0), 2))
        df = df.withColumn('perc_event_view',       F.round(when(col('count_event_type_idx_history__view') > 0, (col('count_event_type_idx_history__view'))/(col('count_event_type_idx_history__view')+col('count_event_type_idx_history__search'))).otherwise(0), 2))

        return df

    def filter(self, df):
        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()
        print(init_timestamp, max_timestamp)

        # Filter minin interactions
        df_item    = df.groupBy("ItemID").count()
        df_item    = df_item.filter(col('count') >= self.minimum_interactions)
        print("Filter minin interactions", df_item.count())

        # Filter session size
        df_session    = df.groupBy("SessionID").count()
        df_session    = df_session.filter(col('count') >= self.min_session_size)
        print("Filter session size", df_session.count())

        df = df \
            .join(df_item, "ItemID", how="inner") \
            .join(df_session, "SessionID", how="inner")

        return df

    def add_available_items(self, df):
        all_items = list(df.select("ItemID").dropDuplicates().toPandas()["ItemID"])

        df = df.withColumn('AvailableItems', udf_sample_items(all_items, self.size_available_list)(col("ItemID")))

        return df

    def fillna(self, df):
        # fill
        df = df.fillna(0, subset=['category_idx', 'condition', "domain_idx", "product_id", "item_id"])\
                .fillna("[]", subset=['title'])\
                .fillna(0.0, subset=['price'])      


        return df

    def vectorize_dense(self, df):

        # for c in self.columns_vectorize_dense:
        #     df  = df.withColumn(c, col(c).cast(FloatType()))

        assembler = VectorAssembler(
            inputCols=self.columns_vectorize_dense,
            outputCol="dense_features",
            handleInvalid="keep")

        scalerizer=StandardScaler().setInputCol("dense_features").setOutputCol("scaled_dense_features")

        df = assembler.transform(df)        
        df = scalerizer.fit(df).transform(df)

        return df

    def input_path(self):
        return self.input()[0].path

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        '''
        Dense Features
        '''
        self.columns_vectorize_dense = ['step', 'cum_Timestamp',
                                        'count_mode_category_idx_history', 'count_mode_condition_idx_history', 
                                        'count_mode_domain_idx_history', 'count_mode_product_id_history', 'count_mode_event_type_idx_history', 
                                        'count_event_type_idx_history__search', 'count_event_type_idx_history__view', 
                                        'count_condition_idx__new', 'count_condition_idx__used', 
                                        'last_price_norm', 'diff_price_norm', 'min_last_price_norm', 'max_last_price_norm', 'mean_last_price_norm', 'sum_last_price_norm', 
                                        'perc_newlest_search', 'perc_event_view']

        spark    = SparkSession(sc)

        # Item Metadada
        df_item  = spark.read.option("delimiter", ",").csv(self.input()[3].path, header=True, inferSchema=True)\
                    .fillna(0, subset=['category_idx', 'condition', "domain_idx", "product_id"])\
                    .fillna(0.0, subset=['price'])      

        # Session
        df      = spark.read.option("delimiter", ",").csv(self.input_path(), header=True, inferSchema=True)
        df      = df.withColumnRenamed("session_id", "SessionID")\
                    .withColumnRenamed("event_timestamp", "Timestamp")\
                    .withColumnRenamed("event_view", "ItemID")\
                    .withColumn("ItemID", col("ItemID").cast("int"))\
                    .withColumn("Timestamp", col("Timestamp").cast("timestamp"))\
                    .orderBy(col('Timestamp'), col('SessionID'))\
                    .select("SessionID", "ItemID", "Timestamp", 
                            "event_type_idx", "event_search", "domain_count", "item_id_count")#.sample(fraction=0.01)#.limit(1000)

        # Add new information
        df = df.withColumn('event_type_click',   when(col('event_type_idx') == ML_SEARCH, 0).otherwise(1))


        #df = df.orderBy(col('Timestamp'), col('SessionID'))
        #df = df.orderBy(col('Timestamp').desc()).dropDuplicates(['SessionID', 'ItemID', 'event_type_idx', 'event_search']) #TODO  ajuda ou atrapalha?
        


        if not self.no_filter_data:
            # Drop duplicate item in that same session
            df = df.dropDuplicates(['SessionID', 'Timestamp', 'event_type_idx']) 
            
            # Filter 
            df = self.filter(df)#.cache()
        
        # Join session with item metadada
        df = df.join(df_item, df.ItemID == df_item.item_id, how="left")\
        
        # Fillna
        df = self.fillna(df)


        # Add more information
        df = self.add_more_information(df)
        
        # add lag variable
        df = self.add_history(df)

        # Add more Information after lag
        df = self.add_more_information_after_history(df)

        #df = self.add_available_items(df)
        # Remove Search event

        df = df.filter(df.event_type_idx != ML_SEARCH) # "search"

        if self.no_filter_data:
            df = df.filter(df.event_type_idx == ML_BUY) #buy
        
        #if not self.no_filter_data:
            # Drop duplicate item in that same session
        #    df = df.dropDuplicates(['SessionID', 'ItemID', 'event_type_idx', 'event_search']) #TODO  ajuda ou atrapalha?

        # vectorize dense features
        #df = self.vectorize_dense(df)
        # cast
        df = df.dropDuplicates(['SessionID', 'ItemID', 'event_type_idx', 'event_search', 'last_event_type_idx'])
        df = df.filter((df.event_type_idx == ML_BUY) | (df.last_ItemID != df.ItemID)) # TODO drop last_ItemID -> ItemID
        

        df = df.cache()

        df = df.withColumn("last_ItemID", df.last_ItemID.cast(IntegerType()))
        df = df.withColumn("product_id", df.product_id.cast(IntegerType()))
        df = df.withColumn("last_event_search", df.last_event_search.cast(StringType()))
        df = df.withColumn("last_title_search", df.last_title_search.cast(StringType()))
        

        df = df.orderBy(col("SessionID")).cache()
        
        df.select(DATA_COLUMNS).sample(fraction=1.0 if DEBUG else 0.01).toPandas().to_csv(self.output()[0].path, index=False)
        deDupeDfCols(df, "_").write.parquet(self.output()[1].path)

class SessionPrepareTestDataset(SessionPrepareDataset):
    sample_days: int = luigi.IntParameter(default=365)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=1)
    min_interactions: int = luigi.IntParameter(default=0)
    min_session_size: int = luigi.IntParameter(default=0)
    no_filter_data: bool = luigi.BoolParameter(default=True)

    def input_path(self):
        return self.input()[1].path

    def requires(self):
        return TextDataProcess()

class SessionPrepareLocalTestDataset(SessionPrepareDataset):
    sample_days: int = luigi.IntParameter(default=365)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=1)
    min_interactions: int = luigi.IntParameter(default=0)
    min_session_size: int = luigi.IntParameter(default=0)
    no_filter_data: bool = luigi.BoolParameter(default=True)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared__test_sample={}_win={}_list={}_min_i={}_min_s={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions, self.min_session_size),)),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared__test_sample={}_win={}_list={}_min_i={}_min_s={}.parquet"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions, self.min_session_size),))                    

    def input_path(self):
        return self.input()[2].path

    def requires(self):
        return TextDataProcess()

class SessionInteractionDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    days_test: int = luigi.IntParameter(default=1)
    index_mapping_path: str = luigi.Parameter(default=None)
    filter_only_buy: bool = luigi.BoolParameter(default=False)
    sample_view: int = luigi.IntParameter(default=0)
    min_interactions: int = luigi.IntParameter(default=5)
    min_session_size: int = luigi.IntParameter(default=2)
    normalize_dense_features: str = luigi.Parameter(default="min_max")
    normalize_file_path: str = luigi.Parameter(default=None)


    def requires(self):
        return SessionPrepareDataset(sample_days=self.sample_days, 
                                    history_window=self.history_window, 
                                    size_available_list=self.size_available_list,
                                    minimum_interactions=self.min_interactions,
                                    min_session_size=self.min_session_size)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def stratification_property(self) -> str:
        return "ItemID"

    @property
    def item_property(self) -> str:
        return "ItemID"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[1].path

    @property
    def task_name(self):
        return self.task_id.split("_")[-1]

    @property
    def scaler_file_path(self):
        if self.normalize_file_path != None:
            return DATASET_DIR+'/'+self.normalize_file_path
        return DATASET_DIR+'/{}_std_scaler.pkl'.format(self.task_name)

    def transform_all(self, df):
        # Cast List()
        columns = [ "title_search_history"]           
        for c in columns:
            if c in df.columns:
                df[c] = df[c].progress_apply(lambda l: [list(i) for i in l])

        columns = [ "last_ItemID_title",
                    "step_history",     
                    "timestamp_history",                      
                    "cum_timestamp_history",                  
                    "event_type_idx_history",                 
                    "category_idx_history",                   
                    "condition_idx_history",                  
                    "domain_idx_history",                     
                    "product_id_history",                     
                    "price_history",                          
                    "ItemID_history"]         
        
        for c in columns:
            if c in df.columns:
                df[c] = df[c].progress_apply(list)
        #from IPython import embed; embed()
        # Cast Int/Str
        columns_int = ["last_ItemID", "mode_category_idx_history", "mode_condition_idx_history",
                        "mode_domain_idx_history", "mode_product_id_history", "mode_event_type_idx_history"]
        for c in columns_int:
            if c in df.columns:
                df[c] = df[c].progress_apply(lambda x: 0 if x == '' else x).fillna(0).progress_apply(int).progress_apply(str)

        # Cast Int
        columns_int = ["last_ItemID"]
        for c in columns_int:
            if c in df.columns:
                df[c] = df[c].fillna(0).progress_apply(int)

        #df['visit'] = 1


    def build_dense_features(self, df: pd.DataFrame, data_key: str):
        columns_vectorize_dense = ['step', 'cum_Timestamp',
                                    'count_mode_category_idx_history', 'count_mode_condition_idx_history', 
                                    'count_mode_domain_idx_history', 'count_mode_product_id_history', 
                                    'count_mode_event_type_idx_history', 'count_event_type_idx_history__search', 
                                    'count_event_type_idx_history__view', 'count_condition_idx__new', 'count_condition_idx__used', 
                                    'last_price_norm', 'diff_price_norm', 'min_last_price_norm', 
                                    'max_last_price_norm', 'mean_last_price_norm', 'sum_last_price_norm', 
                                    'perc_newlest_search', 'perc_event_view']

        if data_key == 'TRAIN_DATA': 
            if self.normalize_dense_features == "standard":
                self.scaler = StandardScaler()                    
            elif self.normalize_dense_features == "min_max":
                self.scaler = MinMaxScaler()

            self.scaler.fit(df[columns_vectorize_dense])
            pickle.dump(self.scaler, open(self.scaler_file_path,'wb'))

        self.scaler = pickle.load(open(self.scaler_file_path,'rb'))
        #from IPython import embed; embed()
        #df[columns_vectorize_dense] = self.scaler.transform(df[columns_vectorize_dense]).round(3)
        df['dense_features'] = list(np.around(self.scaler.transform(df[columns_vectorize_dense]), 3))
        df['dense_features'] = df['dense_features'].apply(list)

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_parquet(self.read_data_frame_path, columns=DATA_COLUMNS)

        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        if len(df) > 0:
            self.transform_all(df)
            self.build_dense_features(df, data_key)

        if data_key == 'TEST_GENERATOR': 
            df = df[df['event_type_idx'] == ML_BUY] # buy

        elif self.filter_only_buy:
            if self.sample_view > 0:
                _val_size = 1.0/self.n_splits if self.dataset_split_method == "k_fold" else self.val_size

                if data_key == 'VALIDATION_DATA':
                    _sample_view_size = int(self.sample_view * _val_size)
                else:
                    _sample_view_size = int(self.sample_view * (1-_val_size))
                
                df_buy  = df[df['event_type_idx'] == ML_BUY] # buy
                df_view = df[df['event_type_idx'] != ML_BUY].sample(_sample_view_size, random_state=42) # view

                df = pd.concat([df_buy, df_view])
            else:
                df = df[df['event_type_idx'] == ML_BUY] # buy

        return df

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

