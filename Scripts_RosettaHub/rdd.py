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

def file_to_rdd(file):
    """
    This function takes a file name and converts it into an RDD.

    Arguments:
    file (str): file name

    Returns:
    An RDD containing all information extracted from the file.
    """

    if file[-3:] == "csv" :
        data = spark.read.format("csv").option("inferSchema", "true")\
                                        .option("delimiter", ',')\
                                        .option("header", 'true')\
                                        .load(file).cache()
        adj_cache = data.persist()

        rdd = adj_cache.rdd.map(tuple)
        return rdd

    elif file[-3:] == "txt" :
        rdd_web = sc.textFile(file) \
                    .map(lambda line: line.split('\t')) \
                    .filter(lambda line: len(line)>1) \
                        .map(lambda line: (line[0],line[1]))

        return rdd_web


def CCF_DEDUP_rdd(rdd):
    """
    This function takes an RDD and returns for each component the closest
    connected neighbor, in one way or the other: we can have (a,b) or (b,a), or both.
    It is inspired from the CCF_iterate and and the CCF_Dedup found
    in the article https://www.cse.unr.edu/~hkardes/pdfs/ccf.pdf

    Arguments:
    rdd (rdd): rdd name

    Returns:
    An RDD containing for each component the closest connected
    neighbor(direct relationship only),in one way or the other.
    """

    # Our goal is to list all existing edges in both ways: (k,v) and (v,k)
    # Our called RDD contains all (k,v) and we want to add all (v,k)

    rdd_reverse = rdd.map(lambda x :(x[1], x[0])) # getting all (v,k)
    rdd_0 = rdd.union(rdd_reverse) # Building a new RDD containing all (k,v) and (v,k)

    # Grouping by key on the first element (k, [v1, v2...])
    rdd_1 = rdd_0.groupByKey().map(lambda x : (x[0], list(x[1])))

    # New k: the minimum between k and all elements included in v
    # New v: all values from k and v
    rdd_2 = rdd_1.map(lambda x : (min(x[0], min(x[1])),  list(set(x[1] + [x[0]]))))

    # Extracting each element of v as our key k and assigning it the corresponding minimum found above
    rdd_3 = rdd_2.flatMapValues(lambda x : x).map(lambda x : (x[1], x[0])).distinct()

    return rdd_3

def get_groups_rdd(rdd):

    """
    This function extracts connected components from an RDD, and assigns
    the smallest component value of each group as the group name.

    Arguments:
    rdd (rdd): rdd name

    Returns:
    t (float) : Computational Time
    size (int) : Total number of distinct edges in the input graph
    num_of_groups (int) : Number of groups of connected components
    rdd (rdd) : An RDD containing as many tuples as the number of unique components in
    the original RDD: the key is the component and the value is the group name.
    """

    #Count of edges
    rdd_reverse = rdd.map(lambda x :(x[1], x[0])) # getting all (v,k)
    rdd_0 = rdd.union(rdd_reverse) # Building a new RDD containing all (k,v) and (v,k)
    size = (rdd_0.distinct()).count()/2

    # Final number of tuples must be equal to the number of distinct values in our original RDD
    # And we can prove that they are equal only once the solution is found
    t = time.time()

    #The counter counts the number of times the algorithm goes through the loop
    #It will depend on the maximum distance between two components of the same group
    counter = 0
    while rdd.count()!= (rdd.groupBy(lambda x : x[0]).distinct()).count() :

        counter +=1
        rdd = CCF_DEDUP_rdd(rdd) # function explained above
        rdd = rdd.persist() # to fight the laziness of PySpark
    t = time.time() - t

    #Getting the number of groups of connected components
    num_of_groups = len(rdd.values().distinct().collect())

    return t, size, num_of_groups, counter, rdd

def main() :
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    rdd = file_to_rdd(input_path)
    t, size, num_of_groups, counter,  rdd = get_groups_rdd(rdd)

    #Save grouped components to file
    rdd.coalesce(1).saveAsTextFile(output_path)

    print("\n\n\n")
    print("The program using DataFrame starts here : ")
    print("---------------------------------------- \n")
    print('There are' + str(num_of_groups) +  'of groups of components in the graph ' + (input_path))
    print('It took ' + (t) + ' seconds to compute the number of groups in this graph of ' + (size) + ' edges.')
    print("\n\n\n\n")



if __name__ == '__main__' :
    main()
