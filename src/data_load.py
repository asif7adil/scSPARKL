from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

import pandas as pd
import os

from helper_functions import melt_spark

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def read_pandas(path_to_file):
    '''
    Pandas funtion to read the input cell x gene CSV file and create spark dataframes.

    This function creates TWO output spark dataframmes:
        - One in wide format same as input, and
        - One in tall format with gene columns melted as spark works better with tall format
        - Required column name fixes are applied as well
    
    :param path_to_file: path to input CSV file (cell x gene matrix). e.g. 'C:/data/brain_counts.csv'. First column in the input file should be cell names.

    :return: wide-format spark data-frame `df` and tall-format spark data-frame `df_melt`. Column name fixes applied.
    '''
    df = pd.read_csv(path_to_file)

    # rename first column to 'cell'
    df = df.rename(columns={df.columns[0]:'cell'})

    # remove ., space, (, ) from column names as it causes issues in spark
    cols_rename = {cols:cols.replace('.','_').replace(' ','_').replace('(','_').replace(')','_') \
        for cols in df.columns if '.' in cols or ' ' in cols or '(' in cols or ')' in cols}
    df = df.rename(columns=cols_rename)

    # melt to tall
    df_melt = pd.melt(df,id_vars=['cell'],value_vars=df.columns[1:])

    # convert to spark data-frames
    df = spark.createDataFrame(df)
    df_melt = spark.createDataFrame(df_melt)

    return df, df_melt


def read_spark(path_to_file,write_to_disk=True):
    '''
    Spark function to read the input cell x gene CSV file and create spark dataframes.
    
    This function creates TWO output spark dataframmes:
        - One in wide format same as input, and
        - One in tall format with gene columns melted as spark works better with tall format
        - Required column name fixes are applied as well
    
    Note: this function may not work with less memory. In that case, use read_pandas().

    :param path_to_file: path to input CSV file (cell x gene matrix). e.g. 'C:/data/brain_counts.csv'. First column in the input file should be cell names.
    :param write_to_disk: Boolean determining whether to write df and df_melt as parquet to disk. This breaks lineage from csv file. Data written to 'interim' folder in code directory.

    :return: wide-format spark data-frame `df` and tall-format spark data-frame `df_melt`. Column name fixes applied.
    '''
    if write_to_disk:
        print('write_to_disk is set to True. df and df_melt will be written as parquet to ./interim/')
        try: os.mkdir('interim')
        except: print('  directory `interim` already exists')
    else:
        print('write_to_disk is set to False. df & df_melt will not be saved to disk.')

    print('reading csv now')
    df = spark.read.csv(path_to_file, header=True, inferSchema=True, maxColumns=300000)
    print('csv read completed..starting cleaning')
    
    # rename first column to 'cell'
    df = df.withColumnRenamed(df.columns[0],'cell')

    # remove ., space, (, ) from column names as it causes issues in spark (invalid characters: " ,;{}()\\n\\t=")
    cols_rename = [cols.replace('.','_').replace(' ','_').replace('(','_').replace(')','_') for cols in df.columns]
    df = df.toDF(*cols_rename)

    # remove ., space, (, ) from cell column
    df = df.withColumn('cell',F.regexp_replace(F.col('cell'), '[.\s()]', '_'))
    print('cleaning completed')
    # persist the table; remove this step if it causes memory issues
    #df = df.persist()
    #df.count()
    
    print('melting data started')
    
    # melt to tall
    df_melt = melt_spark(df,id_vars=['cell'], value_vars=df.drop('cell').columns)
    df_melt = df_melt.repartition(200)
    df_melt = df_melt.persist()
    #df_melt.count()
    print('melting completed..moving to saving as parquet')

    if write_to_disk:
        # parquet dump for interim tables
        
        #use this to write files to hdfs if using in distributed mode, replace address and port.
        #df.write.parquet('hdfs://<address:port>/scRNA/interim/df/', mode = 'overwrite')
        
        df.write.parquet('./interim/df/', mode='overwrite')
        print('writing df parquet completed..writing df_melt now')
        
        #use this to write files to hdfs if using in distributed mode
        #df_melt.write.parquet('hdfs://<address:port>/scRNA/interim/df_melt/', mode = 'overwrite')
        
        df_melt.write.parquet('./interim/df_melt/', mode='overwrite')
        print('df_melt parquet completed..now reading')
        
        ###------------------------------------------------read the files-----------------------------####
        
        df= spark.read.parquet('./interim2/df/')
        #df= spark.read.parquet('hdfs://<address:port>/scRNA/interim/df/')
        
        print('df read operation completed..reading melted df now')
        
        df_melt = spark.read.parquet('./interim/df_melt/')
        
        #use this to write files to hdfs if using in distributed mode
        #df_melt = spark.read.parquet('hdfs://<address:port>/scRNA/interim/df_melt/')

    return df, df_melt


def read_spark_t(path_to_file,write_to_disk=True):
    '''
    Spark function to read the input gene x cell (transposed) CSV file and create spark dataframes.
    
    This function creates TWO output spark dataframmes:
        - One in wide format same as input, and
        - One in tall format with gene columns melted as spark works better with tall format
        - Required column name fixes are applied as well
    
    Note: this function may not work with less memory. In that case, use read_pandas().

    :param path_to_file: path to input transposed CSV file (gene x cell matrix). e.g. 'C:/data/brain_counts.csv'. First column in the input file should be cell names.
    :param write_to_disk: Boolean determining whether to write df and df_melt as parquet to disk. This breaks lineage from csv file. Data written to 'interim' folder in code directory.

    :return: wide-format spark data-frame `df` and tall-format spark data-frame `df_melt`. Column name fixes applied.
    '''
    if write_to_disk:
        print('write_to_disk is set to True. df and df_melt will be written as parquet to ./interim/')
        try: os.mkdir('interim')
        except: print('  directory `interim` already exists')
    else:
        print('write_to_disk is set to False. df & df_melt will not be saved to disk.')

    print('reading csv now')
    #df = spark.read.csv(path_to_file, header=True, inferSchema=True, maxColumns=300000)
    df = spark.read.csv(path_to_file, header=True, inferSchema= True, maxColumns=300000)
    print('csv read completed..starting cleaning')
    # rename first column to 'cell'
    df = df.withColumnRenamed(df.columns[0],'gene')

    # remove ., space, (, ) from column names as it causes issues in spark (invalid characters: " ,;{}()\\n\\t=")
    cols_rename = [cols.replace('.','_').replace(' ','_').replace('(','_').replace(')','_') for cols in df.columns]
    df = df.toDF(*cols_rename)
    
    # remove ., space, (, ) from gene column
    df = df.withColumn('gene',F.regexp_replace(F.col('gene'), '[.\s()]', '_'))
    print('cleaning completed')

    # persist the table; remove this step if it causes memory issues
    #df = df.persist()
    #df.count()
    print("melting of data started")

    # melt to tall
    df_melt = melt_spark(df,id_vars=['gene'], value_vars=df.drop('gene').columns)
    df_melt = df_melt.repartition(200)
    df_melt = df_melt.persist()
   #df_melt.count()
    df_melt = df_melt.withColumnRenamed('variable','cell')
    df_melt = df_melt.withColumnRenamed('gene','variable')
    
    print('melting completed..now writing parquet')
    
    if write_to_disk:
        # parquet dump for interim tables
        #df = df.coalesce(1)
        #df.show(1)
        
        #use this to write files to hdfs if using in distributed mode
        #df.write.parquet('hdfs://<address:port>/scRNA/interim/df/', mode = 'overwrite')
        
        df.write.parquet('./interim/df/', mode = 'overwrite')
        print('df write parquet completed')
        
        #df_melt = df_melt.coalesce(1)
        
        #use this to write files to hdfs if using in distributed mode
        #df_melt.write.parquet('hdfs://<address:port>/scRNA/interim/df_melt/', mode = 'overwrite')
        
        df_melt.write.parquet('./interim/df_melt/', mode = 'overwrite')
        print('df_melt write parquet completed .. now reading parquet files')
        
        ####-----------------------------------read the files---------------------------------------######
        
        #use this to write files to hdfs if using in distributed mode
        #df= spark.read.parquet('hdfs://<address:port>/scRNA/interim/df/')
        
        df = spark.read.parquet('./interim/df/')
        
        #use this to write files to hdfs if using in distributed mode
        #df_melt = spark.read.parquet('hdfs://<address:port>/scRNA/interim/df_melt/')
        
        
        df_melt = spark.read.parquet('./interim/df_melt/')

    return df, df_melt