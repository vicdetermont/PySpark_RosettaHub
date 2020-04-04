# Regular packages
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

# pyspark packages
from pyspark.sql.functions import col, concat, collect_list,struct
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import udf, array, array_distinct, array_min,array_max,array_union, explode
from pyspark.sql.types import IntegerType, DoubleType, ArrayType

from pyspark import SparkContext
from pyspark.sql import Row

import sys

sc = SparkContext()
spark = SparkSession.builder.appName('abc').getOrCreate()

def file_to_df(file):
    """
    This function takes a file name and converts it into a DataFrame.

    Arguments:
    file (str): file name

    Returns:
    An DataFrame containing all information extracted from the file.
    """

    if file[-3:] == "csv" :
        data = spark.read.format("csv").option("inferSchema", "true")\
                                        .option("delimiter", ',')\
                                        .option("header", 'true')\
                                        .load(file).toDF("To","From").cache()
        adj_cache = data.persist()

        return adj_cache

    elif file[-3:] == "txt" :
        df_web = sc.textFile(file) \
            .map(lambda line: (line.split('\t'))).toDF()\
            .select(col('_1').cast(IntegerType()).alias('To'), col('_2').cast(IntegerType()).alias('From'))

        #Remove header if there
        #df_web = df_web.filter(df_web.To !='FromNodeId' )

        return df_web

def our_union (x):
    x[1].append(x[0])
    return x[1]

our_union_udf = f.udf(our_union, ArrayType(IntegerType()))
findmin = f.udf(lambda x: min(x), IntegerType())
our_distinct = f.udf(lambda x: list(set(x)), ArrayType(IntegerType()))

def CCF_DEDUP_df(df):
    """
    This function takes an DataFrame and returns for each component the closest
    connected neighbor, in one way or the other: we can have (a,b) or (b,a), or both.
    It is inspired from the CCF_iterate and and the CCF_Dedup found
    in the article https://www.cse.unr.edu/~hkardes/pdfs/ccf.pdf

    Arguments:
    df (df): DataFrame name

    Returns:
    An DataFrame containing for each component the closest connected
    neighbor(direct relationship only),in one way or the other.
    """

    # Our goal is to list all existing edges in both ways: (k,v) and (v,k)
    # Our called RDD contains all (k,v) and we want to add all (v,k)
    reverseDF = df.select(col("From").alias("To"),col("To").alias("From"))# getting all (v,k)
    df_0 = df.union(reverseDF)# Building a new DataFrame containing all (k,v) and (v,k)

    # Grouping by key on the first element (k, [v1, v2...])
    df_1 = df_0.groupBy(col("To")).agg(our_distinct(collect_list(col("From"))).alias('From'))

    # New k: the minimum between k and all elements included in v
    # New v: all values from k and v
    #df_2 = df_1.withColumn('From', array_union(df_1.From, array(df_1.To))).withColumn('To', findmin("From"))
    df_2 = df_1.withColumn('From', our_union_udf(struct(df_1.To, df_1.From)))\
                    .withColumn('To', findmin("From"))\
                        .withColumn('From', our_distinct('From'))

    # Extracting each element of v as our key k and assigning it the corresponding minimum found above
    df_3 = df_2.select( explode(col("From")).alias("To"), col("To").alias("From")).dropDuplicates()

    return df_3

def get_groups_df(df):

    """
    This function extracts connected components from a DataFrame, and assigns
    the smallest component value of each group as the group name.

    Arguments:
    df (df): DataFrame name

    Returns:
    t (float) : Computational Time
    size (int) : Total number of distinct edges in the input graph
    num_of_groups (int) : Number of groups of connected components
    df (DataFrame) : An DataFrame containing as many tuples as the number of unique components in
    the original DataFrame: the key is the component and the value is the group name.
    """

    #Count of edges
    reverseDF = df.select(col("From").alias("To"),col("To").alias("From")) # getting all (v,k)
    df_0 = df.union(reverseDF)# Building a new DataFrame containing all (k,v) and (v,k)
    size = df_0.distinct().count()/2

    # Final number of tuples must be equal to the number of distinct values in our original RDD
    # And we can prove that they are equal only once the solution is found
    t = time.time()

    #Counter measures the max distance between two neighboors of the group
    counter = 0
    while df.count()!= df.select('To').distinct().count() :
        counter +=1

        df = CCF_DEDUP_df(df) # function explained above
        df = df.persist() # to fight the laziness of PySpark
    t = time.time() - t

    #Getting the number of groups of connected components
    num_of_groups = len(df.select('From').distinct().collect())

    return t, size,num_of_groups, counter,  df

def main() :
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    df = file_to_df(input_path)
    t, size, num_of_groups, counter, df = get_groups_df(df)

    #Save grouped components to file
    df.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv(output_path)

    print("\n\n\n")
    print("The program using DataFrame starts here : ")
    print("---------------------------------------- \n")
    print('There are' + str(num_of_groups) +  'of groups of components in the graph ' + (input_path))
    print('It took ' + (t) + ' seconds to compute the number of groups in this graph of ' + (size) + ' edges.')
    print("\n\n\n\n")



if __name__ == '__main__' :
    main()
