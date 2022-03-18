# differential expression
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import Window
from scipy.stats import t

def diff_expr_top_n(df_melt_flt, cluster_id=0, n=10):
    '''
    Take filtered tall-format dataframe, cluster_id of interest and n as inputs
    Find top n differentially expressed genes in vs out cluster_id
    Uses p-value from two-tailed Independent two-sample t-test to determine top 10 significant genes.

    Actual genes output as significantly different may be less than n if the 5% significance is not met

    scipy is used to get the final p-value from the estimated t test.

    returns dictionary of top N gene:p-value- {gene1:p-value1, gene2:p-value2, ...}
    '''
    # cluster id marking
    df = df_melt_flt.withColumn('cluster_group',F.when(F.col('prediction')==cluster_id,'in').otherwise('out'))
    # t-test related computations on all genes
    # 1. sample size, mean, std. dev.
    df_ttest = df.groupBy('variable','cluster_group').agg(
        F.count('cell').alias('size'),
        F.mean('value').alias('mean'),
        F.stddev('value').alias('sd'))

    # 2. Pivot in vs out into two columns
    df_ttest_p = df_ttest.groupBy('variable').pivot('cluster_group').agg(
        F.max('size').alias('size'),
        F.max('mean').alias('mean'),
        F.max('sd').alias('sd'))

    # compute interim values and t-test value
    df_ttest_p = df_ttest_p.withColumn('df', F.col('in_size') + F.col('out_size') -2)
    df_ttest_p = df_ttest_p.withColumn(
        'sdp',
        F.sqrt(
            (
                (F.col('in_size')-1)*F.col('in_sd')*F.col('in_sd') +
                (F.col('out_size')-1)*F.col('out_sd')*F.col('out_sd')
            )/F.col('df')
        )
    )
    
    df_ttest_p = df_ttest_p.withColumn(
        't_test',
        (
            F.col('in_mean')-F.col('out_mean')
        )/
        (
            F.col('sdp') * F.sqrt(1/F.col('in_size') + 1/F.col('out_size'))
        )
    )

    # use scipy to get p-value: two-tailed
    p_val =  F.udf(lambda t_val, df: float(t.sf(abs(t_val), df=df)*2), FloatType())

    df_ttest_p = df_ttest_p.withColumn('p_val', p_val(F.col('t_test'), F.col('df')))
    
    df_ttest_p = df_ttest_p.persist()
    df_ttest_p.count()
    
    # filter for 5% significance & sort using p-value to get top N genes
    # p-value can be used directly because we are using p-values from tests on same two sets which means sample sizes and df are same
    df_ttest_p = df_ttest_p.where(F.col('p_val')<=0.05)

    df_ttest_p = df_ttest_p.withColumn('rank', F.row_number().over(Window.orderBy(F.col('p_val'))))
    df_ttest_p = df_ttest_p.where(F.col('rank')<=n)

    # return dictionary of top N gene:p-value
    gene_dict = df_ttest_p.select('variable','p_val').toPandas()
    gene_dict = dict(zip(gene_dict['variable'],gene_dict['p_val']))

    return gene_dict