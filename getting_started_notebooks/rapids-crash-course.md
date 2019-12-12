# **Learning RAPIDS (A Crash Course Curriculum)**
## Introductions

In this Crash Course, we're going to cover the basic skills you need to accelerate your data analytics and ML pipeline.  We'll cover how to use the libraries cuDF, cuML, cuGraph, and cuXFilter, as well as exosystem partners, like XGBoost, Dask, and BlazingSQL, to accelerate how you: 
- ingest data
- perform your prepare your data with [ETL (Extract, Transform, and Load)](https://www.webopedia.com/TERM/E/ETL.html)
- model, inference, and predict
- visualize your data throughout the process.  

Each section should take you less than 2 hours to complete.  By the time you're done, you should be able to either:
1. take an existing workflow in a data science or ML pipeline and use a RAPIDS to accelerate it with your GPU
1. Create your own workflows from scratch

This Crash Course was written with the expectation that you know Python, Jupyter Lab.  It is helpful, but not necessary, to have atleast some understanding of Pandas, Scikit Learn, NetworkX, and Datashader. 

[You should be able to run these exercises and use these libraries on any machine with these prerequisties](https://rapids.ai/start.html#PREREQUISITES), which namely are 
- OS of Ubuntu 16.04 or 18.04 or CentOS7 with gcc 5.4 & 7.3
- an NVIDIA GPU of Pascal Architeture or better (basically 10xx series or newer)

RAPIDS does work on Consumer GPUs
## Titan RTX
- [NVIDIA Spot on Titan RTX and RAPIDS](https://www.youtube.com/watch?v=tsWPeZTLpkU)
- [t-SNE 600x Speed up on Titan RTX](https://www.youtube.com/watch?v=_4OehmMYr44)

## Other hardware
- [RAPIDS workflow on an $800 MSI Laptop](https://www.youtube.com/watch?v=7Bw1OqVuLtQ)

Let's get started!

## **1. The Basics of RAPIDS: cuDF, Dask, and cuMl Supervised Learning**
### Introduction
cuDF and cuML are the two fundamental libraries in RAPIDS.  cuDF lets you create and manipulate your dataframes.  It's API is designed to be similar to Pandas.  cuML lets you run many common algorithms and methods on your dataframe so that you can model, infer, regress, reduce, and predict outcomes.  It's API is similar to Scikit Learn.  

Sometimes the dataframe is larger than your available GPU memory.  Dask is used to help our algorithms scale up and using distributed computing.  Using Dask for your distributed computing calculations, whether you have a single GPU, multiple GPUs, or multiple nodes with signle or multiple GPUs, you can process your dataframe no matter the size, just like a regular CPU cluster.  Unfortunately, Dask won't work on Colab, so you need to provision your own machine.

Let's get started with a couple videos!

### Videos

- [Video- Getting Started with RAPIDS](https://www.youtube.com/watch?v=T2AU0iVbY5A).  
 Walks through the [01_Introduction_to_RAPIDS](getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb) notebook which shows, at a high level, what each of the packages in RAPIDS are as well as what they do.
- [Video - RAPIDS: Dask and cuDF NYCTaxi Screencast](https://www.youtube.com/watch?v=gV0cykgsTPM)

  Shows you have you can use RAPIDS and Dask to easily ingest and model a large dataset (1 year's worth of NYCTaxi data) and then create a model around the question "when do you get the best tips".  This same workload can be done on any GPU.

### Learning Notebooks

| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01_Introduction_to_RAPIDS](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb)  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.  |                                                                                                                                    
| [02_Introduction_to_cuDF](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/02_Introduction_to_cuDF.ipynb)  | This notebook shows how to work with cuDF DataFrames in RAPIDS.                                                                                                                                      |
| [03_Introduction_to_Dask](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/03_Introduction_to_Dask.ipynb)   | This notebook shows how to work with Dask using basic Python primitives like integers and strings.                                                                                                                                      |
| [04_Introduction_to_Dask_using_cuDF_DataFrames](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)   | This notebook shows how to work with cuDF DataFrames using Dask.                                                                                                                                      |
| [06_Introduction_to_Supervised_Learning](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/06_Introduction_to_Supervised_Learning.ipynb)   | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.                                                                                                                                      |



### Extra credit and Exercises
- [1- Review of cuDF](https://github.com/rapidsai/notebooks-contrib/blob/master/conference_notebooks/SCIPY_2019/cudf/01-Intro_to_cuDF.ipynb)

- [2- Creating User Defined Fuctions (UDFs) in cuDF](https://github.com/rapidsai/notebooks-contrib/blob/master/conference_notebooks/SCIPY_2019/cudf/02-Intro_to_cuDF_UDFs.ipynb)


## **2. Scaling out and Starting with Algorithms: XGBoost and cuML Dimensionality Reductions and Clustering**
### Introduction
Congrats learning the basics of cuDF, cuML, and Dask.  Now that you have an understanding of how to perform ETL on your data and then do some regression,  regularization, and some modelling, let's look at how further on how to increase your speed up with [XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/), scale it out with Dask XGboost, then see how to use cuML for Dimensionality Reduction and Clustering.

Let's look at a few video walkthroughs of XGBoost, as it may be an unfarmilar concept to some, and then expereince how to use the above in your learning notebooks.  

### Videos
- [Video - Introduction to XGBoost](https://www.youtube.com/watch?v=EQR3bP6XFW0)

 Walks through the [07_Introduction_to_XGBoost](getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb) notebook and shows how to work with GPU accelerated XGBoost in RAPIDS.

- [Video - Introduction to Dask XGBoost](https://www.youtube.com/watch?v=q8HfEZythjM).
 
 Walks through the [08_Introduction_to_Dask_XGBoost](getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb) notebook and hows how to work with Dask XGBoost in RAPIDS.  This can be run on a Single GPU as well and is usefull when your dataset is larger than the memory size of your GPU.

### Learning Notebooks

| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [07_Introduction_to_XGBoost](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb)   | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.                                                                                                                                      |
| [08_Introduction_to_Dask_XGBoost](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb)   | This notebook shows how to work with Dask XGBoost in RAPIDS.                                                                                                                                      | 
| [09_Introduction_to_Dimensionality_Reduction](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/09_Introduction_to_Dimensionality_Reduction.ipynb)   | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.                                                                                                                                      |
| [10_Introduction_to_Clustering](https://github.com/rapidsai/notebooks-contrib/blob/master/getting_started_notebooks/intro_tutorials/10_Introduction_to_Clustering.ipynb)  | This notebook shows how to do GPU accelerated Clustering in RAPIDS. |


### Extra credit and Exercises
- [1- Intro to Linear Regression](https://github.com/rapidsai/notebooks-contrib/blob/master/conference_notebooks/SCIPY_2019/cuml/01-Introduction-LinearRegression-Hyperparam.ipynb)

- [2- Logistic Regression](https://github.com/rapidsai/notebooks-contrib/blob/master/conference_notebooks/SCIPY_2019/cuml/02-LogisticRegression.ipynb)

- [3- Intro to UMAP](https://github.com/rapidsai/notebooks-contrib/blob/master/conference_notebooks/SCIPY_2019/cuml/03-UMAP.ipynb)

### Conclusion to Sections 1 and 2
Here ends the basics of cuDF, cuML, Dask, and XGBoost.  These are libraries that everyone who uses RAPIDS will go to every day.  Our next sections will cover libraries that are more niche in usage, but are powerful to accomplish your analytics.  

## **3. Graphs on RAPIDS: Intro to cuGraph**

Graphs is an exteremely popular area of analytics.  It helps netflix recommend shows, Google rank their sites, connects bits of discrete knowledge into a comprehensive corpus, schedules NFL games, and can even help you optomize seating for your wedding (and it works too!).  [KDNuggests has a great in depth guide to graphs here](https://www.kdnuggets.com/2017/12/graph-analytics-using-big-data.html).  But it was a painfully expensive and slow operation as you added more notes and edges.

[RAPIDS' cuGraph library makes graph analytics effotless, as it boasts some of our best speedups](https://www.zdnet.com/article/nvidia-rapids-cugraph-making-graph-analysis-ubiquitous/), (up to 25,000x).  To put it in persepctive, what can take over 20 hours, cuGraph can lets you do in less than a minute (3 seconds).  In this section, we'll look at some examples of cuGraph methods for your graph analytics and look at a simple use case.

| Topic          | Notebook                                                     | Description                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Centrality      |                                                              |                                                              |
|                 | [Katz](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/centrality/Katz.ipynb)                                | Compute the Katz centrality for every vertex                 |
| Community       |                                                              |                                                              |
|                 | [Louvain](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/community/Louvain.ipynb)                           | Identify clusters in a graph using the Louvain algorithm     |
|                 | [Spectral-Clustering](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/community/Spectral-Clustering.ipynb)   | Identify clusters in a  graph using Spectral Clustering with both<br> - Balanced Cut<br> - Modularity Modularity |
|                 | [Subgraph Extraction](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/community/Sungraph-Extraction.ipynb)   | Compute a subgraph of the existing graph including only the specified vertices |
|                 | [Triangle Counting](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/community/Triangle-Counting.ipynb)       | Count the number of Triangle in a graph                      |
| Components      |                                                              |                                                              |
|                 | [Connected Components](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/components/ConnectedComponents.ipynb) | Find weakly and strongly connected components in a graph     |
| Core            |                                                              |                                                              |
|                 | [K-Core](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/cores/kcore.ipynb)                                  | Extracts the K-core cluster                                  |
|                 | [Core Number](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/cores/core-number.ipynb)                       | Computer the Core number for each vertex in a graph          |
| Link Analysis   |                                                              |                                                              |
|                 | [Pagerank](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/link_analysis/Pagerank.ipynb)                     | Compute the PageRank of every vertex in a graph              |
| Link Prediction |                                                              |                                                              |
|                 | [Jacard Similarity](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/link_prediction/Jaccard-Similarity.ipynb) | Compute vertex similarity score using both:<br />- Jaccard Similarity<br />- Weighted Jaccard |
|                 | [Overlap Similarity](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/link_prediction/Overlap-Similarity.ipynb) | Compute vertex similarity score using the Overlap Coefficient |
| Traversal       |                                                              |                                                              |
|                 | [BFS](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/traversal/BFS.ipynb)                                   | Compute the Breadth First Search path from a starting vertex to every other vertex in a graph |
|                 | [SSSP](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/traversal/SSSP.ipynb)                                 | Single Source Shortest Path  - compute the shortest path from a starting vertex to every other vertex |
| Structure       |                                                              |                                                              |
|                 | [Renumbering](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/structure/Renumber.ipynb) <br> [Renumbering 2](structure/Renumber-2.ipynb) | Renumber the vertex IDs in a graph (two sample notebooks)    |
|                 | [Symmetrize](https://github.com/rapidsai/notebooks/branch-0.11/cugraph/structure/Symmetrize.ipynb)                     | Symmetrize the edges in a graph                              |


### Experiencing the Speedups
### Some End to End examples
| Notebook Name | Description | Colab (if available) |
| ---- | ---- | ---- |


## **4. Integrating RAPIDS with Other Tools**





## **Other Tutorials**
### Towards Data Science

- [GPU Accelerated Data Analytics & Machine Learning](https://towardsdatascience.com/gpu-accelerated-data-analytics-machine-learning-963aebe956ce) - By Pier Paolo

 - Video: https://www.youtube.com/watch?v=LztHuPh3GyU

 - [cuDF, cuML notebook (Colab)](https://drive.google.com/open?id=1oEoAxBbZONUqm4gt9w2PIzmLTa7IjjV9)
 - [cuGraph notebook (Colab)](https://drive.google.com/open?id=1cb40U3gdXZ7ASQsWZypzBFrrFOKpvnbB)
 - [Dask notebook (Colab)](https://drive.google.com/open?id=1jrHoqh_zH7lIsWNsyfRaq0aUARkkW1s2)


- [Deep Learning Analysis Using Large Model Support](https://towardsdatascience.com/deep-learning-analysis-using-large-model-support-3a67a919255) - By Pier Paolo
 - [Large Model Support Notebook: Keras Introduction (Colab)](https://drive.google.com/open?id=1_y81JZWOh4nWUxiY3eaO4FVpxBCB1uN1)
