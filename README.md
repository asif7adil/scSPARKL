# scSPARKL
### An Apache Spark based computational work for the downstream analysis of scRNA-seq data.


## Description
scSPARKL is a simple framework for conducting variety of analysis on scRNA-seq data. It works on the Apache Spark environment which can either be standalone or in distributed mode (depending up on dataload). 


## Prerequisites
Current implementation has been tested on Microsoft Windows Operating System. 
Other Important prerequisites include:
<br>
[Java (latest)](https://www.java.com/download/ie_manual.jsp)
<br>[Apache Spark 3.0 or latest](https://archive.apache.org/dist/spark/)
<br>[Python 3.9 or latest](https://www.python.org/downloads/)
<br>[Jupyer Notebook (Optional)](https://jupyter.org/install)


## Installation
To install the Apache Spark on Windows use any of the following links:
<br>
[Spark Standalone on Windows](https://medium.com/analytics-vidhya/installing-and-using-pyspark-on-windows-machine-59c2d64af76e)
<br>[Apache Spark on Windows](https://dev.to/awwsmm/installing-and-running-hadoop-and-spark-on-windows-33kc)


## Spark Memory allocations
Apache spark is distributed in-memory analytics engine. It is highly recommended to efficiently determine the number of cores and tune the memory alocations to the executors and drivers. To effectively utilize the power of Apache Spark follow the tutorial below for memory configuration and assigning executor cores:
<br>
<br>[Cofigure Executors and Drivers in Spark](https://spoddutur.github.io/spark-notes/distribution_of_executors_cores_and_memory_for_spark_application.html)


## Implementation
Download the source code and run either the Jupyter Notebook or the scSPARKL script file.
Following are the main tasks performed by the pipeline:

### Data Melting
Input data is first cleaned and melted to tall format. 

### Generate Cell, Gene Quality Summaries and filtering the unwanted cells and genes.
This is followed by generating the variety of quality summaries for genes and cells, output of which is saved in an analyses folder automatically generated. 
The quality summaries are then passed as arguments to the `data_filter()` to filter out the unwanted genes and cells. Defualt paremeters can be changed by directly manipulating the filter package.
Additionally new columns can be added for other operations for filtering.

### Normalization
Normalization is currently based on two types:
- Global or Simple normalization; which is similar to CPM normalization.
- Quantile Normalization
Normalization takes tall formated data and returns one wide formatted and a tall formated normalized data.

### Selecting Highly Variable Genes
We have two methods for selecting HVG:
- Median Absolute Deviation (MAD). Default Threshold `k` > 3.
- Coefficient of Variance Squared. Returning Top `n` genes.

### Dimension Reduction using PCA
The dimension reduction is performed using PCA.
The PCA implementation is exclusively spark based.

### Read paper for further details


