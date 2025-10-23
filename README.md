# Overview

A useful statistic in cybersecurity is the observation that most companies detect data breaches and attacks roughly 6 months after these events occur. As a result, it would be highly useful to have a trained model to detect attacks that scans activity at least monthly to prevent extended periods of vulnerability. For this project I used the [LANL Multisource Cybersecurity Dataset](https://csr.lanl.gov/data/cyber1/). The dataset is massive and contains multiple log files (auth, dns, flow, etc.) over the course of a 58 consecutive day recording period with labelled attacks from red team activity within the window. Detection models were trained on attack events and tested on randomly selected windows of user activity that included both attacks and normal operations.

## Methods

The dataset contains 1.6+ billion authentication events and memory constraints required that I subsampled only 3.5 million. I took 6 hour windows sampled across the entire 58 day period, maintaining about a 5% attack event presence in the final data sample. This ensured that the random samples contained the lead up time to an attack and the post-attack phase, while not over/undersampling which could bias model training. 

Feature extraction/engineering was performed after some preprocessing (removing service accounts, empty data cells, etc.). 78 features were extracted from broad categories such as: computer access, authentication, behavior, anomalies, sequence, network and graph features. Feature validation was performed, primarily to determine highly correlated feature pairs.

For model training, most temporal features actually had to be dropped as the size of the datasample caused a bias in the models due to data leakage. I trained an isolation forest, random forest, and LSTM to perform predictions on whether an event was an attack event or normal operations, and calculated F1, accuracy, ROC-AUC, recall, and precision metrics. Then an ensemble model was constructed as a simple weighted sum of the model predictions. In practice, the ensemble model would be used in conjunction with all of the other models to flag logs as potentially containing attack events, in which case an investigator could examine the logs with more direct precision.

## Results
![Results](https://github.com/DylanLe37/APT-Detection-Project/blob/main/Images/modelPerformance.png)


The LSTM performs very well (84% accuracy and 0.94 auc) due to both its ability to read in sequence data and the fact that LSTMs are just much stronger models than the isolation and random forest approaches. This means the LSTM can detect attacks extremely reliably with a very reasonable false positive rate, so as not to flood security analysts with too many spurious alerts.

The random forest also performed quite well on the task, though the poor performance of the isolation forest model does bring down the performance of the ensemble model (which in this case is just a weighted combination of the isolation+random forest). Likely this is because isolation forest is ill-suited to the task of detecting APTs, because their activity is not as easily separated from normal user behavior.

### Model Behavior Analysis

<div align="center">
  <img src="Images/LSTM ROC.png" width="45%" />
  <img src="Images/LSTM Confusion Matrix.png" width="45%" />
</div>

The above are the ROC curve and confusion matrix for the LSTM after training. The ROC curve shows that the LSTM is incredibly robust across a range of thresholds making it a good choice for deployment generally. The confusion matrix is shown as a percentage rather than raw values, and we can see that the false positives make up <0.5% of total events as well as false negatives taking up a very small (<1%) proportion as well. This means that we are unlikely to overwhelm our security team with alerts, and would miss (in principle) very few events from the LSTM alone. The false negatives can be mitigated by employing multiple models trained on different types of training data so as to catch any holes left open by the LSTM's sequence based detection.

<div align="center">
  <img src="Images/Random Forest ROC.png" width="45%" />
  <img src="Images/Random Forest Confusion Matrix.png" width="45%" />
</div>

Similar to the LSTM case, the ROC curve for the random forest model shows impressive performance over a range of thresholds, though notably so compared to the LSTM. Regardless, the false negatives and false positive rate for this model is still quite low and would be manageable if deployed in an active setting. The greatest value of this model, however, is that in context of the previously described LSTM, it is not trained on the same type of data. The events used to train the random forest are not sequence based so by employing both models together, we are likely to detect attack events that neither model could do alone.

## Concluding Remarks
Using relatively simple models (random forest + a pretty shallow LSTM) we can achieve very high performance in detecting APTs. This enables fast analysis of auth logs to in an enterprise setting to prevent long lag times between when attacks occur and when the victim becomes aware of it. Some rough estimates when examining the LSTM false positive rate of 0.35% indicates that, for a  team of 5 analysts that sees an average of 10,000 events/day, each analyst would only need examine 70 events/day in a "normal", non-compromised setting.

### How to run
You can just clone the repo and add the LANL dataset path to the line 71 in main.py.

### Technologies Used
- **Python 3.8** - Primary language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Random Forest, Isolation Forest
- **TensorFlow/Keras** - LSTM implementation
- **Matplotlib/Seaborn** - Visualization
