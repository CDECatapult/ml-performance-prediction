# performance-prediction
Code related to the paper **Predicting the Computational Cost of Deep Learning Models**. This code allows to predict the execution time for commonly used layers within deep neural networks - and, by combining these, for the full network.

This is intended as starting point for an open source machine learning tool, capable of accurately predicting the time that is required to train any neural network on any given hardware. As such, it is easy for everyone to add additional hardware, model layers, or input features, or optimise the prediction model itself.

The folder *models* contains code for benchmarking deep neural networks as well as single layers within these.

The folder *build prediction_model* contains code to generate training data for the model described in the above paper, a data preparation pipeline, and the model training procedures. This folder also contains the training data and the existing tensorflow models.

The folder *prediction* contains the trained models and tools to infer the execution time of arbitrary models on GPUs.
