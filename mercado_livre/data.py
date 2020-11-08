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

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "mercado_livre", "dataset")

BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "train_dataset.jl")
BASE_TEST_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "test_dataset.jl")
BASE_METADATA_FILE : str = os.path.join(OUTPUT_PATH, "mercado_livre", "mercado_livre", "item_data.jl")

## AUX
pad_history = F.udf(
    lambda arr, size: list(reversed(([0] * (size - len(arr[:-1][:size])) + arr[:-1])))[:size], 
    #lambda arr, size: list((arr[:size] + [0] * (size - len(arr[:size])))), 
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

from dateutil.parser import parse
from pyspark.sql.types import TimestampType
parse_date =  udf (lambda x: parse(x), TimestampType())

class PreProcessSessionDataset(BasePySparkTask):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_dataset.csv"))
    
    def get_path_dataset(self):
        return BASE_DATASET_FILE

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark    = SparkSession(sc)
        df = spark.read.json(self.get_path_dataset())

        if not 'item_bought' in df.columns:
            df = df.withColumn('item_bought', lit(0))

        df = df.withColumn("session_id", F.monotonically_increasing_id())

        df = df.withColumn("event", explode(df.user_history))

        df = df.withColumn('event_info', col("event").getItem("event_info"))\
                .withColumn('event_timestamp', col("event").getItem("event_timestamp"))\
                .withColumn('event_type', col("event").getItem("event_type"))
        
        df_view = df.select("session_id", "event_timestamp", "event_info", "event_type")

        df_buy  = df.groupBy("session_id").agg(max(df.event_timestamp).alias("event_timestamp"), 
                                            max(df.item_bought).alias("event_info"))
        df_buy  = df_buy.withColumn('event_type', lit("buy"))
        df_buy  = df_buy.withColumn('event_timestamp', F.date_add(df_buy['event_timestamp'], 1))

        df = df_view.union(df_buy)
        df = df.withColumn('event_timestamp2', parse_date(col('event_timestamp')))

        df.orderBy(col('event_timestamp2')).toPandas().to_csv(self.output().path, index=False)

class PreProcessSessionTestDataset(PreProcessSessionDataset):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset.csv"))
    
    def get_path_dataset(self):
        return BASE_TEST_DATASET_FILE

################################## Supervised ######################################
def char_encode(text):
    vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    return [vocabulary.index(c)+1 if c in vocabulary else 0 for c in unidecode(text).lower() ]

udf_char_encode = F.udf(char_encode, ArrayType(IntegerType()))

class TextDataProcess(BasePySparkTask):
    def requires(self):
        return PreProcessSessionDataset(), PreProcessSessionTestDataset()


    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_dataset__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_dataset__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "item__processed.csv")),\
            luigi.LocalTarget(os.path.join(DATASET_DIR,  "text_vocabulary.csv"))

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
        
        # Add step
        # df = df.withColumn("step", lit(1))

        # w = (Window.partitionBy('session_id').orderBy('event_timestamp2')
        #      .rangeBetween(Window.unboundedPreceding, 0))
        # df = df.withColumn('step', F.sum('step').over(w))

        # add price



        # Split event_info
        df = df.withColumn("event_search",
            when(df.event_type == "search", col("event_info")).
            otherwise(""))

        df = df.withColumn("event_view",
            when(df.event_type == "view", col("event_info")).
            otherwise(0))

        # Add Time
        # w = Window.partitionBy('session_id').orderBy('event_timestamp2')
        # df = df.withColumn("previous_t", lag(df.event_timestamp2, 1).over(w))\
        #         .withColumn("diff_event_timestamp2", (unix_timestamp(df.event_timestamp2) - unix_timestamp(col('previous_t')).cast("integer"))) \
        #         .fillna(0, subset=['diff_event_timestamp2'])                    

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
        df_train = spark.read.option("delimiter", ",").csv(self.input()[0].path, header=True, inferSchema=True)
        df_train = df_train.withColumn("idx", F.monotonically_increasing_id())
        df_train = self.add_more_information(df_train)

        #df_train_text  = df_train.select(["event_search", "idx"]).toPandas()

        # Load train
        df_test = spark.read.option("delimiter", ",").csv(self.input()[1].path, header=True, inferSchema=True)
        df_test = df_test.withColumn("idx", F.monotonically_increasing_id())
        df_test = self.add_more_information(df_test)

        #df_test_text  = df_test.select(["event_search", "idx"]).toPandas()

        # vocabulario
        # vocab = ["<none>"]
        # df_text["title"] = df_text["title"].fillna("<none>")
        # vocab += df_text["title"].tolist() + df_train_text["event_search"].tolist() + df_test_text["event_search"].tolist()

        # Tokenizer
        # tokenizer = StaticTokenizerEncoder(vocab, 
        #     tokenize= self.func_tokenizer, min_occurrences=10, 
        #     reserved_tokens=['<pad>', '<unk>'], padding_index=0)
        # df_vocabulary = pd.DataFrame(tokenizer.vocab, columns=['vocabulary'])
        # print(df_vocabulary)
        # print("Transform Interactions data...")

        #Apply tokenizer 

        ## Metadada
        text_column = "title"
        df = df.withColumn(text_column, udf_char_encode(col(text_column)))
        
        # df_text[text_column] = tokenizer.batch_encode(df_text[text_column])[
        #     0].cpu().detach().numpy().tolist()
        # df_text[text_column + '_max_words'] = len(df_text[text_column][0])

        # df_text = spark.createDataFrame(df_text)
        # df = df.drop(*["title"]).join(df_text, ['item_id'])

        ## Train
        text_column = "event_search"
        df_train = df_train.withColumn(text_column, udf_char_encode(col(text_column)))        
        # df_train_text[text_column] = tokenizer.batch_encode(df_train_text[text_column])[
        #     0].cpu().detach().numpy().tolist()
        # df_train_text[text_column + '_max_words'] = len(df_train_text[text_column][0])

        # df_train_text = spark.createDataFrame(df_train_text)
        # df_train = df_train.drop(*[text_column]).join(df_train_text, ['idx'])

        ## Test
        text_column = "event_search"
        df_test = df_test.withColumn(text_column, udf_char_encode(col(text_column)))                
        # text_column = "event_search"
        # df_test_text[text_column] = tokenizer.batch_encode(df_test_text[text_column])[
        #     0].cpu().detach().numpy().tolist()
        # df_test_text[text_column + '_max_words'] = len(df_test_text[text_column][0])

        # df_test_text = spark.createDataFrame(df_test_text)
        # df_test = df_test.drop(*[text_column]).join(df_test_text, ['idx'])

        # Save
        df_train.orderBy(col('event_timestamp2')).toPandas().to_csv(self.output()[0].path, index=False)
        df_test.orderBy(col('event_timestamp2')).toPandas().to_csv(self.output()[1].path, index=False)
        df.toPandas().to_csv(self.output()[2].path, index=False)
        #df_vocabulary.to_csv(self.output()[3].path)

        return

class SessionPrepareDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    min_session_size: int = luigi.IntParameter(default=2)
    no_filter_data: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return PreProcessSessionDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions),))


    def add_more_information(self, df):
        
        # Add Last
        w = Window.partitionBy('SessionID').orderBy('Timestamp')
        df = df.withColumn("last_ItemID", lag(df.ItemID, 1).over(w))\


        # Add step
        df = df.withColumn("step", lit(1))

        w = (Window.partitionBy('SessionID').orderBy('Timestamp')
             .rangeBetween(Window.unboundedPreceding, 0))
        df = df.withColumn('step', F.sum('step').over(w))

        # Add Time diff and cum
        w = Window.partitionBy('SessionID').orderBy('Timestamp')
        df = df.withColumn("previous_t", lag(df.Timestamp, 1).over(w))\
                .withColumn("diff_Timestamp", (unix_timestamp(df.Timestamp) - unix_timestamp(col('previous_t')).cast("integer"))) \
                .fillna(0, subset=['diff_Timestamp'])           
        
        df = df.withColumn('int_Timestamp', df.Timestamp.cast("integer"))

        w = Window.partitionBy('SessionID').orderBy(col("Timestamp"))
        df = df.withColumn('cum_Timestamp', F.sum('diff_Timestamp').over(w)).fillna(0, subset=['cum_Timestamp'])      

        return df

    def add_history(self, df):
        
        w = Window.partitionBy('SessionID').orderBy('Timestamp')#.rowsBetween(Window.unboundedPreceding, Window.currentRow)#.rangeBetween(Window.currentRow, 5)

        # History Step
        df = df.withColumn(
            'StepHistory', F.collect_list('step').over(w)
        )
        df = df.withColumn('StepHistory', pad_history(df.StepHistory, lit(self.history_window)))


        # History Time
        df = df.withColumn(
            'TimestampHistory', F.collect_list('int_Timestamp').over(w)
        )
        df = df.withColumn('TimestampHistory', pad_history(df.TimestampHistory, lit(self.history_window)))


        w = Window.partitionBy('SessionID').orderBy('Timestamp')#.rangeBetween(Window.currentRow, 5)

        # History Item
        df = df.withColumn(
            'ItemIDHistory', F.collect_list('ItemID').over(w)
        ).where(size(col("ItemIDHistory")) >= self.min_session_size)#\

        df = df.withColumn('ItemIDHistory', pad_history(df.ItemIDHistory, lit(self.history_window)))

        return df

    def filter(self, df):
        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        print("filter date", df.count())
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()
        print("filter date", df.count())
        print(init_timestamp, max_timestamp)

        # Filter minin interactions
        df_item    = df.groupBy("ItemID").count()
        df_item    = df_item.filter(col('count') >= self.minimum_interactions)
        print("Filter minin interactions", df_item.count())

        # Filter session size
        df_session    = df.groupBy("SessionID").count()
        df_session    = df_session.filter(col('count') >= self.min_session_size)
        print("Filter session size", df_session.count())

        print(df.show(50))

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

        spark    = SparkSession(sc)
        df = spark.read.option("delimiter", ",").csv(self.input().path, header=True, inferSchema=True)
        df = df.withColumnRenamed("session_id", "SessionID")\
            .withColumnRenamed("event_timestamp2", "Timestamp")\
            .withColumnRenamed("event_info", "ItemID")\
            .withColumn("ItemID", col("ItemID").cast("int"))\
            .withColumn("Timestamp", col("Timestamp").cast("timestamp"))\
            .orderBy(col('Timestamp'), col('SessionID'))\
            .select("SessionID", "ItemID", "Timestamp", "event_type").limit(1000)


        if not self.no_filter_data:
            # Remove Search event
            df = df.filter(df.event_type != "search")

            # Drop duplicate item in that same session
            df = df.dropDuplicates(['SessionID', 'ItemID', 'event_type']) #TODO  ajuda ou atrapalha?
            
            # Filter 
            df = self.filter(df)

        df = self.add_more_information(df)
        df = self.add_history(df)
        #df = self.add_available_items(df)
        df = df.withColumn('visit',lit(1))
        
        if self.no_filter_data:
            df = df.filter(df.ItemID == 0)

        df.orderBy(col("SessionID")).toPandas().to_csv(self.output().path, index=False)

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

    def requires(self):
        return PreProcessSessionTestDataset()

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

