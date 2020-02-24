# **Learning RAPIDS (A Crash Course Curriculum)**
## Introductions

In this Crash Course, we're going to cover the basic skills you need to accelerate your data analytics and ML pipeline.  We'll cover how to use the libraries cuDF, cuML, cuGraph, and cuXFilter, as well as exosystem partners, like XGBoost, Dask, and BlazingSQL, to accelerate how you: 
- Ingest data
- Perform your prepare your data with [ETL (Extract, Transform, and Load)](https://www.webopedia.com/TERM/E/ETL.html)
- Run modelling, inferencing, and predicting algorithms on the data in a GPU dataframe
- Visualize your data throughout the process.  

Each section should take you less than 2 hours to complete.  By the time you're done, you should be able to:
1. Take an existing workflow in a data science or ML pipeline and use a RAPIDS to accelerate it with your GPU
1. Create your own workflows from scratch

This Crash Course was written with the expectation that you know Python, Jupyter Lab.  It is helpful, but not necessary, to have atleast some understanding of Pandas, Scikit Learn, NetworkX, and Datashader. 

[You should be able to run these exercises and use these libraries on any machine with these prerequisties](https://rapids.ai/start.html#PREREQUISITES), which namely are 
- OS of Ubuntu 16.04 or 18.04 or CentOS7 with gcc 5.4 & 7.3
- an NVIDIA GPU of Pascal Architeture or better (basically 10xx series or newer)

RAPIDS works on Consumer GPUs (like GeForce, Titan), not just Prosumer/Enterprise Class GPUS (Quadro, Tesla, DGX)
## Titan RTX
- [NVIDIA Spot on Titan RTX and RAPIDS](https://www.youtube.com/watch?v=tsWPeZTLpkU)
- [t-SNE 600x Speed up on Titan RTX](https://www.youtube.com/watch?v=_4OehmMYr44)

## Other hardware
- [RAPIDS workflow on an $800 MSI Laptop](https://www.youtube.com/watch?v=7Bw1OqVuLtQ)

Let's get started!

## **1. The Basics of RAPIDS: cuDF and Dask**
### Introduction
cuDF is the fundamental library in RAPIDS.  cuDF lets you create and manipulate your dataframes, which all other libraries use to model, infer, regress, reduce, and predict outcomes. It's API is designed to be similar to Pandas.  

Sometimes the dataframe is larger than your available GPU memory.  Dask is used to help our algorithms scale up and using distributed computing.  Whether you have a single GPU, multiple GPUs, or multiple nodes with single or multiple GPUs, you can use Dask for your distributed computing calculations andorchstrate the processing of your GPU dataframe, no matter the size, just like a regular CPU cluster.  Unfortunately, Dask won't work on Colab, so you need to provision your own machine, like an [AWS dask Cluster]() (*link to blog coming soon*).

Let's get started with a couple videos!

### Videos

| Video Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Video- Getting Started with RAPIDS](https://www.youtube.com/watch?v=T2AU0iVbY5A).  | Walks through the [01_Introduction_to_RAPIDS](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb) notebook which shows, at a high level, what each of the packages in RAPIDS are as well as what they do. |
| [Video - RAPIDS: Dask and cuDF NYCTaxi Screencast](https://www.youtube.com/watch?v=gV0cykgsTPM) | Shows you have you can use RAPIDS and Dask to easily ingest and model a large dataset (1 year's worth of NYCTaxi data) and then create a model around the question "when do you get the best tips".  This same workload can be done on any GPU. |

### Learning Notebooks

If you don't have a GPU enabled system, you can try these notebooks in our [Section 1 Colab](https://colab.research.google.com/drive/10ZMf5DA9GxkVlJ-z39YW9vCoTTky_tDr)

| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01_Introduction_to_RAPIDS](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/01_Introduction_to_RAPIDS.ipynb)  | This notebook shows at a high level what each of the packages in RAPIDS are as well as what they do.  |                                                                                                                                    
| [02_Introduction_to_cuDF](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/02_Introduction_to_cuDF.ipynb)  | This notebook shows how to work with cuDF DataFrames in RAPIDS.                                                                                                                                      |
| [03_Introduction_to_Dask](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/03_Introduction_to_Dask.ipynb)   | This notebook shows how to work with Dask using basic Python primitives like integers and strings.                                                                                                                                      |
| [04_Introduction_to_Dask_using_cuDF_DataFrames](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb)   | This notebook shows how to work with cuDF DataFrames using Dask.                                                                                                                                      |
| [Guide to UDFs](https://github.com/rapidsai/cudf/blob/branch-0.12/docs/cudf/source/guide-to-udfs.ipynb) | This notebook provides and overview of User Defined Functions with cuDF |



### Extra credit and Exercises
- [10 minute review of cuDF](https://rapidsai.github.io/projects/cudf/en/0.12.0/10min-cudf-cupy.html)
- [Extra Credit - 10 minute guide to cuDF and cuPY](https://rapidsai.github.io/projects/cudf/en/0.12.0/10min-cudf-cupy.html)
- [Extra Credit - Multi-GPU with Dask-cuDF](https://rapidsai.github.io/projects/cudf/en/0.12.0/dask-cudf.html)
- [Review and Exercises 1- Review of cuDF](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/conference_notebooks/SCIPY_2019/cudf/01-Intro_to_cuDF.ipynb)
- [Review and Exercises 2- Creating User Defined Fuctions (UDFs) in cuDF](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/conference_notebooks/SCIPY_2019/cudf/02-Intro_to_cuDF_UDFs.ipynb)


## **2. Accelerating those Algorithms: cuML and XGBoost**
### Introduction
Congrats learning the basics of cuDF and Dask.  Now let's take a look at cuML

cuML lets you run many common algorithms and methods on your dataframe so that you can model, infer, regress, reduce, and predict outcomes on the data in your cuDF data frames. It's API is similar to Scikit Learn.  [Among the ever growing suite of algorithms, you can perform several GPU accelerated algortihms for each of these methods:]()(*will link to slide deck showing cuML algorithms*)

- Classification / Regression
- Inference
- Clustering
- Decomposition & Dimensionality Reduction
- Time Series

While we look at cuML , we'll take a look at how further on how to increase your speed up with [XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/), scale it out with Dask XGboost, then see how to use cuML for Dimensionality Reduction and Clustering.

Let's look at a few video walkthroughs of XGBoost, as it may be an unfarmilar concept to some, and then expereince how to use the above in your learning notebooks.  

### Videos

| Video Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Video - Introduction to XGBoost](https://www.youtube.com/watch?v=EQR3bP6XFW0) | Walks through the [07_Introduction_to_XGBoost](getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb) notebook and shows how to work with GPU accelerated XGBoost in RAPIDS. |
| [Video - Introduction to Dask XGBoost](https://www.youtube.com/watch?v=q8HfEZythjM) |  Walks through the [08_Introduction_to_Dask_XGBoost](getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb) notebook and hows how to work with Dask XGBoost in RAPIDS.  This can be run on a single GPU as well and is useful when your dataset is larger than the memory size of your GPU. |

### Learning Notebooks

| Notebook Title         | Description |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [06_Introduction_to_Supervised_Learning](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/06_Introduction_to_Supervised_Learning.ipynb)   | This notebook shows how to do GPU accelerated Supervised Learning in RAPIDS.                                                                                                                                      |
| [07_Introduction_to_XGBoost](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/07_Introduction_to_XGBoost.ipynb)   | This notebook shows how to work with GPU accelerated XGBoost in RAPIDS.                                                                                                                                      |
| [08_Introduction_to_Dask_XGBoost](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/08_Introduction_to_Dask_XGBoost.ipynb)   | This notebook shows how to work with Dask XGBoost in RAPIDS.                                                                                                                                      | 
| [09_Introduction_to_Dimensionality_Reduction](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/09_Introduction_to_Dimensionality_Reduction.ipynb)   | This notebook shows how to do GPU accelerated Dimensionality Reduction in RAPIDS.                                                                                                                                      |
| [10_Introduction_to_Clustering](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/getting_started_notebooks/intro_tutorials/10_Introduction_to_Clustering.ipynb)  | This notebook shows how to do GPU accelerated Clustering in RAPIDS. |


### Extra credit and Exercises
- [10 Review of Dask XGBoost](https://rapidsai.github.io/projects/cudf/en/0.12.0/dask-xgb-10min.html)

- [Review and Exercises 1 - Linear Regression](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/conference_notebooks/SCIPY_2019/cuml/01-Introduction-LinearRegression-Hyperparam.ipynb)

- [Review and Exercises 2 -  Logistic Regression](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/conference_notebooks/SCIPY_2019/cuml/02-LogisticRegression.ipynb)

- [Review and Exercises 3- Intro to UMAP](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.12/conference_notebooks/SCIPY_2019/cuml/03-UMAP.ipynb)

### RAPIDS cuML Example Notebooks
- [Index of Notebooks](https://github.com/rapidsai/notebooks#cuml-notebooks)
- [Direct Link to Notebooks](https://github.com/rapidsai/notebooks/tree/branch-0.12/cuml)


### Conclusion to Sections 1 and 2
Here ends the basics of cuDF, cuML, Dask, and XGBoost.  These are libraries that everyone who uses RAPIDS will go to every day.  Our next sections will cover libraries that are more niche in usage, but are powerful to accomplish your analytics.  

## **3. Graphs on RAPIDS: Intro to cuGraph**

It is often useful to look at the relationships contained in the data, which we do that thought the use of graph analytics. Representing data as a graph is an extremely powerful techniques that has grown in popularity.  Graph analytics are used to helps Netflix recommend shows, Google rank sites in their search engine, connects bits of discrete knowledge into a comprehensive corpus, schedules NFL games, and can even help you optimize seating for your wedding (and it works too!). [KDNuggests has a great in depth guide to graphs here](https://www.kdnuggets.com/2017/12/graph-analytics-using-big-data.html).  Up until now, running a graph analytics was a painfully slow, particularly as the size of the graph (number of nodes and edges) grew.

[RAPIDS' cuGraph library makes graph analytics effotless, as it boasts some of our best speedups](https://www.zdnet.com/article/nvidia-rapids-cugraph-making-graph-analysis-ubiquitous/), (up to 25,000x).  To put it in persepctive, what can take over 20 hours, cuGraph can lets you do in less than a minute (3 seconds).  In this section, we'll look at some examples of cuGraph methods for your graph analytics and look at a simple use case.

### RAPIDS cuML Example Notebooks
- [Index of Notebooks](https://github.com/rapidsai/notebooks/#cugraph-notebooks)
- [Direct Link to Notebooks](https://github.com/rapidsai/notebooks/tree/branch-0.12/cugraph)


### Experiencing the Speedups
### Some End to End examples
| Notebook Name | Description | Colab (if available) |
| ---- | ---- | ---- |


## **4. Integrating RAPIDS with Other Tools**




## External Tutorials:
### Bringing in Deep Learning by "Towards Data Science"

[GPU Accelerated Data Analytics & Machine Learning](https://towardsdatascience.com/gpu-accelerated-data-analytics-machine-learning-963aebe956ce) - By Pier Paolo

 - Video: https://www.youtube.com/watch?v=LztHuPh3GyU

 - [cuDF, cuML notebook (Colab)](https://drive.google.com/open?id=1oEoAxBbZONUqm4gt9w2PIzmLTa7IjjV9)
 - [cuGraph notebook (Colab)](https://drive.google.com/open?id=1cb40U3gdXZ7ASQsWZypzBFrrFOKpvnbB)
 - [Dask notebook (Colab)](https://drive.google.com/open?id=1jrHoqh_zH7lIsWNsyfRaq0aUARkkW1s2)


[Deep Learning Analysis Using Large Model Support](https://towardsdatascience.com/deep-learning-analysis-using-large-model-support-3a67a919255) - By Pier Paolo
 - [Large Model Support Notebook: Keras Introduction (Colab)](https://drive.google.com/open?id=1_y81JZWOh4nWUxiY3eaO4FVpxBCB1uN1)
