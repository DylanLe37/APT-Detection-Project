# Overview

A useful statistic in cybersecurity is the observation that most companies detect data breaches and attacks roughly 6 months after these events occur. As a result, it would be highly useful to have a trained model to detect attacks that scans activity at least monthly to prevent extended periods of vulnerability. For this project I used the [LANL Multisource Cybersecurity Dataset]([url](https://csr.lanl.gov/data/cyber1/)). The dataset is massive and contains multiple log files (auth, dns, flow, etc.) over the course of a 58 consecutive day recording period with labelled attacks from red team activity within the window. Detection models were trained on attack events and tested on randomly selected windows of user activity that included both attacks and normal operations.

## Methods

The dataset contains 1.6+ billion authentication events and memory constraints required that I subsampled only 3.5 million. I took 6 hour windows sampled across the entire 58 day period, maintaining about a 5% attack event presence in the final data sample. This ensured that the random samples contained the lead up time to an attack and the post-attack phase, while not over/undersampling which could bias model training. 

Feature extraction/engineering was performed after some preprocessing (removing service accounts, empty data cells, etc.). 78 features were extracted from broad categories such as: computer access, authentication, behavior, anomalies, sequence, network and graph features. Feature validation was performed, primarily to determine highly correlated feature pairs.

For model training, most temporal features actually had to be dropped as the size of the datasample caused a bias in the models due to data leakage. I trained an isolation forest, SVM, random forest, and LSTM to perform predictions on whether an event was an attack event or normal operations, and calculated F1, accuracy, ROC-AUC, recall, and precision metrics. Then an ensemble model was constructed as a simple weighted sum of the model predictions. In practice, the ensemble model would be used in conjunction with all of the other models to flag logs as potentially containing attack events, in which case an investigator could examine the logs with more direct precision.

## Results
