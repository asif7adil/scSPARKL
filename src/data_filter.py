# functions module for data QC and filtering
import os
import pyspark.sql.functions as F

try:
    os.mkdir('file:///D:/scRNA/analyses')
    print('created `analyses` directory for writing qc outputs')
except:
    print('`analyses` directory already exists. This will be used to write qc outputs.')

def qc_matrix_cells(df_melt):
    '''
    Takes tall-format genome dataframe and returns QC summary dataframe for cells.
    QC summary contains:
        - first column of df (name unchanged)
        - sum_of_all_entries: sum of all cells per row
        - average_of_sum: avg of all cells per row
        - total_number_of_columns: count of columns for each row where entry is >0
        - total_number_of_ERCC_entry: count of columns where column name starts with “ERCC” and entry in that is >0
        - sum_of_ercc: sum of all columns where column name starts with “ERCC”
        - percentage_of_ercc: calculated by {(sum_of_ercc/sum_of_all_entries)*100}
        - log1p_of_sum_of_all_entries: log(1 + sum_of_all_entries)
        - log1p_of_sum_of_ercc: log(1 + sum_of_ercc)
        
    :param df_melt: tall-format spark Dataframe. column names should be ['cell','variable','value']. Use read_pandas() or read_spark() to load data
        
    :return: df_agg spark DataFrame containing qc summary.
    Output columns ['cell','sum_of_all_entries','average_of_sum','total_number_of_columns','total_number_of_ERCC_entry','sum_of_ercc','percentage_of_ercc','log1p_of_sum_of_all_entries','log1p_of_sum_of_ercc']
    '''
    # add extra columns for required summary cols; to be aggregated to get final summary
    df_melt = df_melt.withColumn(
        'total_number_of_columns',
        F.when(F.col('value')>0,1).otherwise(0))
    df_melt = df_melt.withColumn(
        'total_number_of_ERCC_entry',
        F.when((F.lower(F.col('variable')).contains('ercc')) & (F.col('value')>0),1).otherwise(0))
    df_melt = df_melt.withColumn(
        'sum_of_ercc',
        F.when(F.lower(F.col('variable')).contains('ercc'),F.col('value')).otherwise(0))
    df_melt = df_melt.withColumn(
        'total_number_of_MT_entry',
        F.when((F.lower(F.col('variable')).contains('mt')) & (F.col('value')>0),1).otherwise(0))
    df_melt = df_melt.withColumn(
        'sum_of_mt',
        F.when(F.lower(F.col('variable')).contains('mt'),F.col('value')).otherwise(0))

    # aggregate to 1 row per cell
    df_agg= df_melt.groupBy('cell').agg(
        F.sum('value').alias('sum_of_all_entries'),
        F.avg('value').alias('average_of_sum'),
        F.sum('total_number_of_columns').alias('total_number_of_columns'),
        F.sum('total_number_of_ERCC_entry').alias('total_number_of_ERCC_entry'),
        F.sum('sum_of_ercc').alias('sum_of_ercc'),
        F.sum('total_number_of_MT_entry').alias('total_number_of_MT_entry'),
        F.sum('sum_of_mt').alias('sum_of_mt'))
    
    # other cols
    df_agg = df_agg.withColumn(
        'percentage_of_ercc', (F.col('sum_of_ercc')/F.col('sum_of_all_entries'))*100)
    df_agg = df_agg.withColumn(
        'percentage_of_mt', (F.col('sum_of_mt')/F.col('sum_of_all_entries'))*100)
    df_agg = df_agg.withColumn(
        'log1p_of_sum_of_all_entries', F.log(1+F.col('sum_of_all_entries')))
    df_agg = df_agg.withColumn(
        'log1p_of_sum_of_ercc', F.log(1+F.col('sum_of_ercc')))
    df_agg = df_agg.withColumn(
        'log1p_of_sum_of_mt', F.log(1+F.col('sum_of_mt')))
    print('cell quality matrix preparation completed')
    #len(df_agg.head(1))>0
    
    return df_agg


def qc_matrix_genes(df_melt):
    '''
    Takes tall-format genome dataframe and returns QC summary dataframe for cells.
    QC summary contains:
        - gene_name: names correspoding to columns headers of input matrix (post special character imputation done in data_load.read_* fns)
        - no_cells_by_counts: count of cells per gene having entry >0
        - sum_of_cells: sum of all entries per gene
        - mean_counts: average of sum_of_cells
        - log1p_of_mean_counts: log(1 + mean_counts)
        - no_of_dropouts: count of cells per gene having entry is ==0
        - pct_dropout_by_counts: percentage of dropouts i.e., [no_of_dropouts/(no_of_dropouts+no_cells_by_counts) * 100]
        - log1p_total_counts: log(1 + sum_of_cells)
        
    :param df_melt: tall-format spark Dataframe. column names should be ['cell','variable','value']. Use read_pandas() or read_spark() to load data
        
    :return: df_agg spark DataFrame containing qc summary.
    Output columns ['gene_name','no_cells_by_counts','sum_of_cells','mean_counts','no_of_dropouts','pct_dropout_by_counts','log1p_of_mean_counts','log1p_total_counts']
    '''
    df_melt = df_melt.withColumnRenamed('variable','gene_name')
    # add extra columns for required summary cols; to be aggregated to get final summary
    df_melt = df_melt.withColumn(
        'no_cells_by_counts',
        F.when(F.col('value')>0,1).otherwise(0))
    df_melt = df_melt.withColumn(
        'no_of_dropouts',
        F.when(F.col('value')==0,1).otherwise(0))

    # aggregate to 1 row per cell
    df_agg= df_melt.groupBy('gene_name').agg(
        F.sum('no_cells_by_counts').alias('no_cells_by_counts'),
        F.sum('value').alias('sum_of_cells'),
        F.avg('value').alias('mean_counts'),
        F.sum('no_of_dropouts').alias('no_of_dropouts'))
    
    # other cols
    df_agg = df_agg.withColumn(
        'pct_dropout_by_counts', (F.col('no_of_dropouts')/(F.col('no_of_dropouts')+F.col('no_cells_by_counts')))*100)
    df_agg = df_agg.withColumn(
        'log1p_of_mean_counts', F.log(1+F.col('mean_counts')))
    df_agg = df_agg.withColumn(
        'log1p_total_counts', F.log(1+F.col('sum_of_cells')))
    print('gene quality matrix preparation completed')
    #len(df_agg.head(1))>0
    
    return df_agg


# updated filter function
def filter_matrix(df_melt, df_qc_cells=None, df_qc_genes=None, apply_ercc=True, apply_mito=False):
    '''
    Takes tall-format genome dataframe, the two QC dataframes and returns tall-format genome dataframe.
    Filters applied:
        If apply_ercc=True:
            - Remove `cells` with percentage_of_ercc>=10%
            - Remove `cells` with less than 10 genes > 0 (total_number_of_columns)
        If apply_mito=True:
            - Remove `cells` with percentage_of_mito>=5%
            - Remove `cells` with less than 10 genes > 0 (total_number_of_columns)
    and/or
        - Remove `genes` with pct_dropout_by_counts>70%
        - Remove `genes` with no_cells_by_counts<3
    Update conditions below to update filter thresholds.

    :param df_melt: tall-format spark Dataframe. column names should be ['cell','variable','value']. Use read_pandas() or read_spark() to load data
    :param df_qc_cells: output dataframe of qc_matrix_cells(). Set to None to NOT apply cells filter
    :param df_qc_genes: None or output dataframe of qc_matrix_genes(). Set to None to NOT apply genes filter
        
    :return: filtered df_melt dataframe
    '''
    if df_qc_cells:
        # Do not remove this step: this sets the base for all other filters to apply
        df_qc_cells = df_qc_cells.withColumn('remove_cells', F.lit(0))
        
        
        # Remaining filtering parameters
        # add parameters according to the need
        df_qc_cells = df_qc_cells.withColumn('remove_cells', F.when(F.col('total_number_of_columns')<10,1).otherwise(F.col('remove_cells')))
        df_qc_cells = df_qc_cells.withColumn('remove_cells', F.when(F.col('sum_of_all_entries')<1000,1).otherwise(F.col('remove_cells')))
        print('first two lines applied')
        # ERCC masking
        if apply_ercc:
            print('apply_ercc set to True, applying filter')
            df_qc_cells = df_qc_cells.withColumn('remove_cells', F.when(F.col('percentage_of_ercc')>=10,1).otherwise(F.col('remove_cells')))
            print('first if executed')
        # mitochondiral filtering parameter
        if apply_mito:
            print('apply_mito set to True, applying filter')
            df_qc_cells = df_qc_cells.withColumn('remove_cells', F.when(F.col('percentage_of_mt')>=5,1).otherwise(F.col('remove_cells')))
            print('2nd if executed')

        # remove this step for saving time consumed by count() and collect()
        #print('cells to be removed:', df_qc_cells.groupBy().agg(F.sum('remove_cells')).collect()[0][0],'of',df_qc_cells.count())
        
        
        # keep only cells where remove_cells is 0
        df_qc_cells = df_qc_cells.where(F.col('remove_cells')==0).select('cell')
    else:
        print('cells filter not applied')

    if df_qc_genes:
        # Do not remove this step: this sets the base for all other filters to apply
        df_qc_genes = df_qc_genes.withColumn('remove_genes', F.lit(0))
        
        df_qc_genes = df_qc_genes.withColumn('remove_genes', F.when(F.col('pct_dropout_by_counts')>90,1).otherwise(F.col('remove_genes')))
        df_qc_genes = df_qc_genes.withColumn('remove_genes', F.when(F.col('no_cells_by_counts')<5,1).otherwise(F.col('remove_genes')))

        #print('genes to be removed:', df_qc_genes.groupBy().agg(F.sum('remove_genes')).collect()[0][0],'of',df_qc_genes.count())
        # keep only genes where remove_genes is 0
        df_qc_genes = df_qc_genes.where(F.col('remove_genes')==0).selectExpr('gene_name as variable')
    else:
        print('genes filter not applied')

    # inner join to df_melt to filter it
    if df_qc_cells:
        df_melt = df_melt.join(df_qc_cells, on=['cell'], how='inner')
    if df_qc_genes:
        df_melt = df_melt.join(df_qc_genes, on=['variable'], how='inner')

    return df_melt