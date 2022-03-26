import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.ml.feature import PCA, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
import collections
import pynndescent

#this line is important for using UMAP with rows >4096
collections.namedtuple("n", [], module=__name__)
pynndescent.rp_trees.FlatTree.__module__ = "pynndescent.rp_trees"



def pca_apply(df_norm,k=2):
    '''
    Takes wide-format genome dataframe (direct or normalized) and returns vectorized PCA components.
    Number of components set via k.
    Prints variance explained by each component.
        
    :param df_norm: wide-format spark Dataframe (cell x gene). column names should be ['cell', gene names...]
        
    :return: df spark DataFrame with vector of PCA components- components
    '''
    # pre-propcess input cols
    col_dtypes = pd.DataFrame(df_norm.drop('cell').dtypes, columns=['col_name','dtype'])

    # Transformation pipeline
    string_cols    = col_dtypes[col_dtypes['dtype']=='string']['col_name'].tolist()
    numerical_cols = col_dtypes[col_dtypes['dtype']!='string']['col_name'].tolist()

    stages = []
    index_cols = []
    if string_cols: # empty list (no string cols) evaluates to False and skips below loop
        for col in string_cols:
            new_col = col + '_index'
            indexer = StringIndexer(inputCol=col, outputCol=new_col, handleInvalid='keep')
            stages.append(indexer)
            index_cols.append(new_col)
    
    print(f'Input has {len(numerical_cols)} numerical cols & {len(index_cols)} string cols.')

    assembler = VectorAssembler(inputCols=numerical_cols + index_cols, outputCol='vec_assemble')
    stages.append(assembler)
    pipe = Pipeline(stages=stages)
    pipe_model = pipe.fit(df_norm)

    df = pipe_model.transform(df_norm)
    df = df.select('cell','vec_assemble')

    # PCA
    pca = PCA(k=k, inputCol="vec_assemble", outputCol="components")
    model = pca.fit(df)
    print('PCA components:',model.getK())
    df = model.transform(df)
    print('explainedVariance')
    print('  By components:', model.explainedVariance)
    print('  Cumulative   :', np.cumsum(model.explainedVariance))
    print('  Total        :', sum(model.explainedVariance))
    
    df = df.select('cell','components')
    return df, model.explainedVariance, np.cumsum(model.explainedVariance)

def umap_apply(df_norm, n_components=2, n_neighbors=15, min_dist=0.5, metric='euclidean'):
    '''
    Takes wide-format genome dataframe (direct or normalized) and returns UMAP components.
    Number of components set via n_components.
    Input is converted to pandas df to use in UMAP and output is converted to spark df before returning.
    Ref: https://umap-learn.readthedocs.io/en/latest/

    :param df_norm: wide-format spark Dataframe (cell x gene). column names should be ['cell', gene names...]
    Should only contain numerical columns apart from 'cell' col
        
    :return: df spark DataFrame with UMAP components
    '''
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    df = df_norm.toPandas()
    embedding = reducer.fit_transform(df.drop('cell',axis=1).values)
    print('embedding shape:', embedding.shape)
    df_umap = pd.DataFrame(data=embedding, columns=range(n_components))
    df_umap['cell'] = df['cell']
    try:
        df_umap.to_csv('./interim/df_umap_temp.csv', index=False)
        df_umap_spark = spark.read.csv('./interim/df_umap_temp.csv', header=True, inferSchema=True)
    except:
        df_umap.to_csv('df_umap_temp.csv', index=False)
        df_umap_spark = spark.read.csv('df_umap_temp.csv', header=True, inferSchema=True)

    return df_umap_spark

def tsne_apply(df_norm, n_components=2, n_iter=2000, perplexity=30, learning_rate=1000, metric='euclidean'):
    '''
    Takes wide-format genome dataframe (direct or normalized) and returns t-sne components.
    Number of components set via n_components.
    Input is converted to pandas df to use in t-sne and output is converted to spark df before returning.

    :param df_norm: wide-format spark Dataframe (cell x gene). column names should be ['cell', gene names...]
    Should only contain numerical columns apart from 'cell' col
        
    :return: df spark DataFrame with t-sne components
    '''
    reducer = TSNE(
        n_components=n_components,
        init='random',
        random_state=0,
        n_iter=n_iter,
        learning_rate=float(learning_rate),
        metric=metric,
        perplexity=float(perplexity),
        method='barnes_hut'
    )
    df = df_norm.toPandas()
    embedding = reducer.fit_transform(df.drop('cell',axis=1).values)
    print('embedding shape:', embedding.shape)
    df_tsne = pd.DataFrame(data=embedding, columns=range(n_components))
    df_tsne['cell'] = df['cell']
    try:
        df_tsne.to_csv('./interim/df_tsne_temp.csv', index=False)
        df_tsne_spark = spark.read.csv('./interim/df_tsne_temp.csv', header=True, inferSchema=True)
    except:
        df_tsne.to_csv('df_tsne_temp.csv', index=False)
        df_tsne_spark = spark.read.csv('df_tsne_temp.csv', header=True, inferSchema=True)

    return df_tsne_spark
