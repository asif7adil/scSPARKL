#!/usr/bin/env python
# coding: utf-8

# Single-Cell RNA-seq Data analysis using scSPARKL- An Apache Spark based parallel computational tool. 
# UMAP is independent of spark environment i.e., it will take it's own good time to run.

import time
import findspark
findspark.init()

import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import scSPARKL modules.
from data_load import read_spark, read_spark_t #use 'read_spark_t' to read a transposed matrix i.e., [gene x cell] type.
from data_filter import qc_matrix_cells, qc_matrix_genes, filter_matrix
from data_normalize import norm_quantile, norm_global
from dimension_reduction import umap_apply, tsne_apply
from clustering_functions import cluster_data_prep, k_means_spark
from differential_expression import diff_expr_top_n
from top_hvg import * 


# #The framework supports `.csv` file formats for now. Also, the input should be in the form of `cell x gene` matrix.


path_to_file = "./data/jurkat_raw_data_annot.csv"



# Use this line to transpose the data
df = pd.read_csv(path_to_file, index_col=0)
df = df.transpose()
df.to_csv(path_to_file.split('.csv')[0]+'_t.csv')


# ### Read new input files
# - Following line reads the files as spark dataframe. The function `read_spark()` performs following tasks:
# - Reads a `.csv` file as a `Spark Dataframe`
# - Performs Cleaning of unknown characters for `spark` and replaces them with an `Underscore`.
# - Melts the dataframe from `wide` fromat to `tall` format.
# - Writes and returns the cleaned dataframe to and as a `parquet` format.

# Completed stages are printed accordingly.
df, df_melt = read_spark(path_to_file)

# Read previously saved parquet data files form the disc storage
get_ipython().run_cell_magic('time', '', "df= spark.read.parquet('./interim/df/')\ndf_melt = spark.read.parquet('./interim/df_melt/')")


# ### Generate Cell and Gene Quality summary
# The output is written as a `.csv` in the `analyses` folder.
# Note: The input is a melted spark dataframe

# Generate Cell Quality Summary
df_qc_cells = qc_matrix_cells(df_melt)\ndf_qc_cells.toPandas().to_csv('./analyses/qc_cells.csv', index=False)
# Generate Gene Quality Summary.
df_qc_genes = qc_matrix_genes(df_melt)\ndf_qc_genes.toPandas().to_csv('./analyses/qc_genes.csv', index=False)


# Plot the summaries for filtering
# convert the spark datframes to pandas to visualize
pd_df_qc_cells = df_qc_cells.toPandas()
pd_df_qc_genes = df_qc_genes.toPandas()

print(pd_df_qc_cells.head(2))
print(pd_df_qc_genes.head(2))

plt.hist(pd_df_qc_cells['total_number_of_columns'], bins=50)
plt.xlabel('N genes')
plt.ylabel('N cells')
plt.axvline(2000, color='black')

plt.hist(pd_df_qc_cells['sum_of_all_entries'], bins=1000)
plt.xlabel('Total counts')
plt.ylabel('N cells')
#plt.axvline(10000, color='red')
plt.xlim(0,1e6)

plt.hist(pd_df_qc_cells['percentage_of_ercc'], bins=50)
plt.xlabel('Percent counts ERCC')
plt.ylabel('N cells')
plt.axvline(10, color='red')


# ### Filter out the unwanted Cells and Genes
# The `filter_matrix()` uses following as the default values for the removal:

# - ERCC percentage > 10%'
# - Mitochondrial percentage > 5%
# - Cells having < 10 genes where count is 1
# - Genes expressing in < 3 cells
# - Dropout rate > 95%
# Note: filtering thresholds can be changed (as desired) in the `data_filter` package

df_melt_flt = filter_matrix(df_melt,df_qc_cells,None, apply_ercc = True, apply_mito = True) #cell filtering applied

df_melt_flt = filter_matrix(df_melt_flt,None,df_qc_genes) #gene filtering applied

# Write filtered matrix to drive as a parquet file.
df_melt_flt.write.parquet('./interim/df_melt_flt/', mode='overwrite')
df_melt_flt = spark.read.parquet('./interim/df_melt_flt/')

# Count Check
{'cells':df_melt_flt.select('cell').drop_duplicates().count(),
'genes':df_melt_flt.select('variable').drop_duplicates().count()}

# # Normalization
# We currently implement two types of normalizations:
# - Quantile Normalization https://doi.org/10.1038/s41598-020-72664-6
# - Global aka simple Normalization aka CPM Normalization
# The output of the normalizations is in `wide` format as well as in `tall` format.

df_norm, df_norm_melt = norm_global(df_melt_flt) #norm_global is similar to cpm normalizaton


#write global normalization
df_norm.write.parquet('./interim/df_norm_G/', mode='overwrite')
df_norm_g = spark.read.parquet('./interim/df_norm_G/')

# Selection of Highly Variable Genes

# There are two methods for selecting top HVGs:
# - Coefficient of Variances squared. Takes 'n' as a parameter, for returning `n` number of HVG genes.
# - Median Absoluute Deviatioin. Takes 'k' as a parameter of threshold

top_hvg = top_hvg(df_norm_melt, calc_cv2=True, n=18000)

# **persist the dataframe and check the count
top_hvg = top_hvg.persist()
top_hvg.count()



# Reduce the dataframe
# Perform PCA & Visualize
# Perform Kmeans based on first 2 PCs
df_pca, pca_components, pca_components_cummulative = pca_apply(top_hvg,k=50)

df_pca.write.parquet('./interim/df_pca/', mode='overwrite')
df_pca = spark.read.parquet('./interim/df_pca/')

# plot PCA Components
x_axis = range(len(pca_components))
y_lim = max(pca_components_cummulative)*1.2

fig, ax = plt.subplots()
ax.bar(x_axis, pca_components, color="C0")
ax.set_ylim([0, y_lim])

ax2 = ax.twinx()
ax2.plot(range(len(pca_components)), pca_components_cummulative, color="C1", marker="D", ms=1)
ax2.set_ylim([0, y_lim])

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()


# extract components- PCA (only required for PCA as PCA output is Spark Vector)
df_plot = df_pca.toPandas()

df_plot['x_comp'] = df_plot['components'].apply(lambda x: x[x_comp])
df_plot['y_comp'] = df_plot['components'].apply(lambda x: x[y_comp])

# plot PCA: update the 'hue' column as required, or skip it if not required
plt.figure(figsize=(15,15))
sns.scatterplot(x=df_plot['x_comp'], y=df_plot['y_comp'], hue=df_plot['sp_nme'])
plt.show()


# ### UMAP & K-means
# Perform UMAP on the data, Visualize and perform Kmeans clustering based on the UMAP embeddings.

#apply UMAP on HVG dataframe\n
df_umap = umap_apply(top_hvg, n_components=2)
# write umap_embeddings to disc
df_umap.write.parquet('./interim/df_umap/', mode='overwrite')
#read parquet UMAP
df_umap = spark.read.parquet('./interim/df_umap/')")


# Join the metadata with UMAP embeddings
# Use these lines for joining the metadata to any of the previously processed data.


df_metadata = spark.read.csv('./data/Jurkat_annotations.csv', header=True, inferSchema=True)


# clean the metadata file
%%time
cols_rename = [cols.replace('.','_').replace(' ','_').replace('(','_').replace(')','_') for cols in df_metadata.columns]
df_metadata = df_metadata.toDF(*cols_rename)

# rename the sample/cells column to cells
%%time
df_metadata = df_metadata.withColumn('cell',F.regexp_replace(F.col('cell'), '[.\s()]', '_'))

# check the count
df_metadata.count()

# perform the join
%%time
df_umap = df_umap.join(df_metadata, on=['cell'],how='left')

# Visualise the UMAP
# indexes for components to visualize
# starts at 0; x_comp=0 and y_comp=1 plots first component on x-axis and second component on y-axis<br>
# update as required
x_comp = 0
y_comp = 1


#extract components from umap
# skip/update the couluring column in the select function (sp_name)
df_plot = df_umap.select(['cell',str(x_comp), str(y_comp), 'sp_nme']).toPandas()

# plot UMAP
plt.figure(figsize=(10,10))
sns.scatterplot(x=df_plot[str(x_comp)], y=df_plot[str(y_comp)], hue = df_plot['sp_nme'])
plt.xlabel('Umap 1')
plt.ylabel('umap 2')
plt.show()


# **KMeans**
# clustering- takes Vector assembled column of features to perform clustering
# (use the cluster_prep fn in case the input features aren't Vector assembled already- only PCA is already vectorized)

df_clust = cluster_data_prep(df_umap.select('cell','0','1'))
df_clust.show(5)


# loop on Kmeans to determine the optimum K
eval_list = []
for k in range(2,20):
    kmeans_model, predictions, eval_metrics =        k_means_spark(df_clust, k=k, getSilhouette=True)
    eval_list.append(eval_metrics)

eval_list = pd.DataFrame(data=eval_list, columns=['k','silhouette','sse'])
eval_list.head()


# plot the elbow curve
plt.figure(figsize=(15,15))
sns.lineplot(x=eval_list['k'], y=eval_list['sse'])
plt.show()

# apply kmeans
kmeans_model, predictions, eval_metrics = k_means_spark(df_clust, k=6)


# checkint the cluster prediction labels added to the data
predictions.show(5)
predictions.groupBy('prediction').count().sort('prediction').show()


# save the prediction labels
predictions.write.parquet('./interim/predictions/', mode='overwrite')
predictions = spark.read.parquet('./interim/predictions/')


# plot the cluster output
x_comp = 0
y_comp = 1

df_plot = df_umap.join(predictions.select('cell','prediction'),on=['cell'],how='left')
df_plot = df_plot.select(['cell',str(x_comp),str(y_comp),'prediction']).toPandas()

plt.figure(figsize=(10,10))
sns.scatterplot(x=df_plot[str(x_comp)], y=df_plot[str(y_comp)], hue=df_plot['prediction'], palette='deep')
plt.show()


#### Differential expression of selected genes across clusters
# this uses post-QC pre-norm data- df_melt_flt

df_melt_flt = spark.read.parquet('./interim/df_melt_flt/')
predictions = spark.read.parquet('./interim/predictions/')

# Add cluster labels
df_melt_flt = df_melt_flt.join(predictions.select('cell','prediction'), on=['cell'],how='left')

# call the DE function for selected cluster id and N
cluster_0_genes = diff_expr_top_n(df_melt_flt,cluster_id=0,n=10)
cluster_1_genes = diff_expr_top_n(df_melt_flt,cluster_id=1,n=10)
cluster_2_genes = diff_expr_top_n(df_melt_flt,cluster_id=2,n=10)
cluster_3_genes = diff_expr_top_n(df_melt_flt,cluster_id=3,n=10)
cluster_4_genes = diff_expr_top_n(df_melt_flt,cluster_id=4,n=10)
cluster_5_genes = diff_expr_top_n(df_melt_flt,cluster_id=5,n=10)
#cluster_6_genes = diff_expr_top_n(df_melt_flt,cluster_id=6,n=10)

# see the top 10 genes for clusters
cluster_0_genes
cluster_0_genes.keys()

cluster_3_genes
cluster_3_genes.keys()

cluster_2_genes
cluster_2_genes.keys()

cluster_1_genes
cluster_1_genes.keys()
