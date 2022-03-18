import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import Window
from scipy.stats import t

from helper_functions import melt_spark


def calc_hvg(df_norm_melt, calc_cv2=True, n=2000, calc_mad=None, k=3):
    '''
    Takes tall-format normalized dataframe and returns dataframe with Highly Variable Gene dataframe.
    Returned datafrmae contains:
        -mean: Equivalent to the RowMeans() of R statistic
        -std: Standard Deviation
        -co_var: coefficient of variance calculated as std/mean.
        -CV2: Coefficient of Variantion squared.
       if cal_mad = True:
           -MAD: Median Absolute Deviation of the gene
           -median: median of the gene
           -median_diff: median difference, calculated as: absolute(gene count value - median)
    
    :param df_norm_melt: tall-format spark Dataframe. Column names should be ['cell','variable','value'].
    :param calc_cv2: Boolean, for calculating the Coefficient of Variance squared. Defualt: True
    :param cal_mad: Boolean, for calculating Median Absolute Deviation(MAD) of each gene. Default: None
    :param n: Integer type, for top n highly hvariable genes.
    :param k: integer type MAD threshold.
    
    :return: df with highly variable genes/MAD filtered genes.
    '''
    # df_norm_melt = df_norm_melt.withColumnRenamed('variable', 'gene_name')
    
    # calculate coefficient of variance squared
    if calc_cv2:
        df_agg = df_norm_melt.groupBy('variable').agg(
            F.mean('norm_value').alias('mean'),
            F.stddev('norm_value').alias('sd'),
            F.variance('norm_value').alias('var'))
    
        # add column for Coefficient of Variation(CV)
        df_agg = df_agg.withColumn(
            'co_var', (F.col('sd')/(F.col('mean'))))
        
        # add column for CV^2
        df_agg = df_agg.withColumn('CV2', pow(F.col('co_var'),2))
        
        # Rank Genes as per the CV^2 column
        df_agg = df_agg.withColumn('rank', F.row_number().over(Window.orderBy(F.col('CV2'))))
        
        df_hvg_filt = hvg_filter(df_agg, n)
    
    # calculate Median Absolute Deviation
    if calc_mad:
        df_hvg_filt = MADf(df_norm_melt, k)
    
    # join hvg/mad matrix with original filtered matrix for filtering out remaining genes
    df_hvg_norm_melt = df_norm_melt.join(df_hvg_filt, on=['variable'], how='inner')
    
    # pivot to wide format
    df = df_hvg_norm_melt.groupBy('cell').pivot('variable').agg(F.max('norm_value'))
    
    return df

def hvg_filter(df_agg, n=2000):
    '''
    Takes wide-format dataframe with co-efficient of variance and rank calculated and returns dataframe filtered with top 'n' Highly Variable Gene dataframe.
    Returned dataframe contains:
        -mean: Equivalent to the RowMeans() of R statistic
        -std: Standard Deviation
        -co_var: coefficient of variance calculated as std/mean.
        -CV2: Coefficient of Variantion squared.
    :param df_agg: wide-format spark Dataframe.
    
    :return: df with top 'n' highly variable genes.
    '''
    df_agg = df_agg.withColumn('remove_genes', F.lit(0))
        
    df_agg = df_agg.withColumn('remove_genes', F.when(F.col('rank')>n,1).otherwise(F.col('remove_genes')))
    
    # keep the genes rank greater than given 'n'
    df_agg = df_agg.where(F.col('remove_genes')==0).selectExpr('variable as variable')
    
    return df_agg

def MADf(df_norm_melt, k=3):
    '''
    Takes tall-format normalized dataframe and returns dataframe with Highly Variable Gene dataframe.
    Returned datafrmae contains:
        -MAD: Median Absolute Deviation of the gene
           -median: median of the gene
           -median_diff: median difference, calculated as: absolute(gene count value - median)
    :param df_norm_melt: tall-format spark Dataframe. Column names should be ['cell','variable','value'].
    :param cal_mad: Boolean argument, for calculating Median Absolute Deviation(MAD) of each gene
    :param k: integer type MAD threshold.
    
    :return: Median Absolute Deviation based filtered dataframe
    '''
    # calculate median of each gene
    median = df_norm_melt.groupBy('variable').agg(F.expr('percentile(norm_value, array(0.5))')[0].alias('median'))
    # print('Median = ',median.show()) 
    
    # join the median to the original df_norm_melt
    df_norm_melt = median.join(df_norm_melt, on=['variable'], how='inner')
        
    # add the column of absolute median difference
    df_norm_melt = df_norm_melt.withColumn("median_diff", F.abs(F.col('norm_value')-F.col('median')))
        
    # find the median of median absolute deviation
    mad = df_norm_melt.groupBy('variable').agg(F.expr('percentile(median_diff, array(0.5))')[0].alias('MAD'))
        
    # join the median absolute deviation with the original df_melt_norm
    df_norm_melt = mad.join(df_norm_melt, on=['variable'], how='inner')
    
    print('calculation of "Median Absolute Deviation" completed..filtering the genes based on k= ',k)
    
    # filter genes with MAD>k
    df_agg = df_norm_melt.withColumn('remove_genes', F.lit(0))
        
    df_agg = df_agg.withColumn('remove_genes', F.when(F.col('MAD')>k,1).otherwise(F.col('remove_genes')))
    
    # keep the genes greater than given 'k'
    df_agg = df_agg.where(F.col('remove_genes')==0).selectExpr('variable as variable')
    
    return df_agg