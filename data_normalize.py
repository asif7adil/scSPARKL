import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
spark.conf.set("spark.sql.pivotMaxValues", 200000)

def norm_quantile(df_melt):
    '''
    Takes tall-format genome dataframe and returns quantile normalized, wide-format genes matrix dataframe.
    Quantile normalization done withtout reference distribution and the values are set to avg. at each rank.
        
    :param df_melt: tall-format spark Dataframe (post filtering). column names should be ['cell','variable','value']
        
    :return: df spark DataFrame with quantile normalized gene values
    '''
    # add the rank cols: rank_compute for getting avgs, rank_assign for assigning norm. vals
    # rank_compute is without any gaps and continous to ensure every rank has values from each gene
    # rank_assign assigns same rank for ties and then skips ranks when moving to next value
    
    # remove ERCC
    df_melt = df_melt.where(~F.lower(F.col('variable')).contains('ercc'))

    df_melt = df_melt.withColumn(
        'rank_compute', F.row_number().over(Window.orderBy('value').partitionBy('variable')))
    df_melt = df_melt.withColumn(
        'rank_assign', F.rank().over(Window.orderBy('value').partitionBy('variable')))
    
    # get avg value per rank
    df_norm = df_melt.groupBy('rank_compute').agg(F.avg('value').alias('norm_value'))
    
    # rename column to use is join later
    df_norm = df_norm.withColumnRenamed('rank_compute','rank_assign')
    df_norm = df_norm.persist()
    df_norm.count()

    # assign back norm values per rank
    df_melt = df_melt.join(df_norm, on=['rank_assign'], how='left')
    df_melt = df_melt.select('cell','variable','norm_value')
    
    # pivot to wide format
    df = df_melt.groupBy('cell').pivot('variable').agg(F.max('norm_value'))

    return df, df_melt


def norm_global(df_melt):
    '''
    Takes tall-format genome dataframe and returns global normalized, wide-format genes matrix dataframe.
    Global normalization is similar to converting to CPM. It dividies each gene value for a cell by sum of all values for that gene.
    This is then multiplied by scaling factor (10,000) and we take log(1+v).
        
    :param df_melt: tall-format spark Dataframe (post filtering). column names should be ['cell','variable','value']
        
    :return: df spark DataFrame with cpm/global normalized gene values
    '''
    # remove ERCC
    # df_melt = df_melt.where(~F.lower(F.col('variable')).contains('ercc'))
    
    # get global sum to use for norm.: sums up all gene values per cell (i.e. sums up columns from the cell x gene matrix)
    df_norm = df_melt.groupBy('cell').agg(F.sum('value').alias('sum_value'))
    df_norm = df_norm.persist()
    df_norm.count()

    # join back norm values
    df_melt = df_melt.join(df_norm, on=['cell'], how='left')
    df_melt = df_melt.withColumn('norm_value', F.log(1+10000*(F.col('value')/F.col('sum_value'))))
    df_melt = df_melt.select('cell','variable','norm_value')
    # pivot to wide format
    df = df_melt.groupBy('cell').pivot('variable').agg(F.max('norm_value'))

    return df, df_melt
