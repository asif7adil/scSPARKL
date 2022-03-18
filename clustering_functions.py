# clustering functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import pandas as pd

def cluster_data_prep(df):
    '''
    takes a cell x gene df where the genes are in multiple columns and returns a vector assembled df
    Vector assembled df is required for k-means model as input

    :param df: wide-format spark Dataframe with normal columns. cell x gene.
               Can contain other features that are to be passed to clustering.
               Can also contain a combination of already assembled vector (e.g. PCA output) and other normal features. The two will then be merged into a single vector for clustering

    :return: df spark DataFrame with 'features' vector that can be given as input to k-means

    '''
    # transformaion pipeline
    col_dtypes = pd.DataFrame(df.drop('cell').dtypes, columns=['col_name','dtype'])

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

    assembler = VectorAssembler(inputCols=numerical_cols + index_cols, outputCol='features')
    stages.append(assembler)
    pipe = Pipeline(stages=stages)
    pipe_model = pipe.fit(df)
    #
    df = pipe_model.transform(df)
    df = df.select('cell','features')
    df = df.persist()
    df.count()

    return df


def k_means_spark(df, k=2, featuresCol='features', predictionsCol='prediction', getSilhouette=False, seed=125):
    """
    pass vector assembled df to get k-means clustering
    Silhoutte score is computed basis getSilhouette param

    returns kmeans_model, predictions df and tuple of (k, silhouette score, sse score)
    """
    #initiate model instance
    kmeans = KMeans(featuresCol=featuresCol, k=k, seed=seed)
    #fitting the model
    kmeans_model = kmeans.fit(df)
    print('k means done')
    #Within Set Sum of Squared Errors
    sse = kmeans_model.summary.trainingCost
    print('sse done')
    # Evaluate clustering by computing Silhouette score
    predictions = kmeans_model.summary.predictions.select(['cell'] + [featuresCol] + [predictionsCol])
    if getSilhouette:
        evaluator = ClusteringEvaluator(featuresCol=featuresCol)
        silhouette = evaluator.evaluate(predictions)
        print('silhouette done')
    else:
        silhouette = 0
        print('silhouette skipped')
    #
    print(k, ": Silhouette = " + str(silhouette), ". SSE = " + str(sse), ".")
    #
    return kmeans_model, predictions, (k,silhouette, sse)
