# Normal vs. Slow Traffic Anomaly Detection Experiment

## 1. Experiment Overview

**Project:** Explainable-AIOps for E-Commerce Applications
**Experiment:** Normal vs. Slow Traffic Comparison
**Model:** LSTM Autoencoder
**Experiment Version:** `normal_extended_v1`
**Evaluation Threshold:** `0.025898804888129234`

### Objective

The objective of this experiment is to evaluate whether an LSTM Autoencoder trained on normal e-commerce traffic can identify potentially anomalous traffic through reconstruction error.

The model reconstructs sequences of traffic observations. A reconstruction error above the configured threshold marks a window as a potential anomaly.

## 2. Model Configuration

The experiment uses the existing LSTM Autoencoder and its associated feature scaler.

| Configuration                     | Value                                                      |
| --------------------------------- | ---------------------------------------------------------- |
| Model                             | LSTM Autoencoder                                           |
| Sequence window size              | 10 observations                                            |
| Number of input features          | 12                                                         |
| Anomaly threshold                 | 0.025898804888129234                                       |
| Threshold selection               | 95th percentile of normal validation reconstruction errors |
| Model retrained during comparison | No                                                         |
| Existing model artifacts modified | No                                                         |

### Input Features

The model uses the following 12 features:

1. `response_time_ms`
2. `log_response_time`
3. `status_200`
4. `status_400`
5. `status_404`
6. `status_other`
7. `endpoint_health`
8. `endpoint_products`
9. `endpoint_latency`
10. `endpoint_other`
11. `is_connection_error`
12. `is_http_error`

Each input window contains 10 observations with 12 features per observation.

## 3. Evaluation Datasets

Two datasets were evaluated using the same saved model, scaler, and threshold.

### 3.1 Normal Held-Out Test Dataset

The normal held-out test dataset contains 81 overlapping sequence windows. It was used to observe how frequently normal test traffic exceeded the anomaly threshold.

### 3.2 Isolated Slow-Traffic Dataset

The isolated slow-traffic dataset contains 40 traffic observations. With a sequence window size of 10, these observations produced 31 overlapping windows for evaluation.

The two datasets differ in size and composition. Therefore, their results should be interpreted as a preliminary comparison rather than a controlled estimate of real-world model performance.

## 4. Experimental Results

| Metric                       |    Normal Held-Out Test |  Isolated Slow Traffic |
| ---------------------------- | ----------------------: | ---------------------: |
| Windows evaluated            |                      81 |                     31 |
| Windows flagged as anomalous |                       1 |                     31 |
| Windows not flagged          |                      80 |                      0 |
| Flagged percentage           |                   1.23% |                100.00% |
| Mean reconstruction error    | Approximately 0.0115278 | Approximately 173.3799 |
| Median reconstruction error  | Approximately 0.0081664 | Approximately 174.0669 |
| Anomaly threshold            |          0.025898804888 |         0.025898804888 |

The displayed error statistics are rounded for readability. The accompanying CSV and JSON files contain the saved evaluation results.

## 5. Results Interpretation

### 5.1 Normal Traffic

One of the 81 normal held-out test windows exceeded the threshold. This corresponds to a flagged-window rate of approximately 1.23% on this dataset.

This result suggests that most evaluated normal windows remained below the configured threshold. However, the result does not establish a general false-positive rate for all normal traffic.

### 5.2 Slow Traffic

All 31 evaluated windows from the isolated slow-traffic dataset exceeded the threshold.

This indicates that the saved model assigned substantially higher reconstruction errors to these evaluated windows than to most windows in the normal held-out test dataset.

The result supports further investigation of reconstruction-error-based anomaly detection. It does not demonstrate that the model will detect every slow-traffic incident in production.

### 5.3 Threshold Interpretation

The threshold was selected using the 95th percentile of reconstruction errors from normal validation data.

The same saved threshold was used for both evaluations. It was not adjusted specifically to improve the slow-traffic result.

## 6. Limitations

The following limitations must be considered:

1. The isolated slow-traffic dataset contains only 40 observations and produces 31 overlapping windows.
2. Overlapping windows are correlated and should not be treated as 31 independent incidents.
3. The datasets differ in size and composition.
4. Results from these datasets do not establish performance on unseen production traffic.
5. The current comparison does not establish precision, recall, F1-score, or generalization performance across diverse anomaly types.
6. Reconstruction-error flags identify potential anomalies; they do not independently determine the operational root cause.
7. The large difference between the reconstruction errors of the two datasets should be investigated to confirm feature scaling, preprocessing consistency, and dataset comparability before broader conclusions are drawn.

## 7. Reproducibility and Evidence

The comparison evidence is stored in:

`ml/evaluation_normal_extended_v1/comparison_report/`

The following files preserve the results:

* `normal_vs_slow_comparison.csv` — tabular comparison metrics.
* `normal_vs_slow_comparison.json` — detailed structured results.
* `normal_vs_slow_comparison.pdf` — formatted experiment report.

The isolated slow-traffic evaluation outputs are stored in:

`ml/evaluation_normal_extended_v1/`

Relevant files include:

* `slow_traffic_results.csv`
* `evaluation_summary.json`

The saved model artifacts are stored in:

`ml/artifacts_normal_extended_v1/`

The comparison was generated without retraining the model or modifying the existing model, scaler, or anomaly threshold.

## 8. Conclusion

In this preliminary experiment, the LSTM Autoencoder flagged 1 of 81 normal held-out test windows and all 31 windows from the isolated slow-traffic dataset.

These findings demonstrate a difference in reconstruction-error behavior between the two evaluated datasets. Further testing with independently collected and labeled normal and anomalous traffic is required before making claims about real-world detection performance.

Future work should include testing additional anomaly types, checking preprocessing consistency, evaluating independent incidents, and calculating appropriate performance metrics using labeled evaluation data.
