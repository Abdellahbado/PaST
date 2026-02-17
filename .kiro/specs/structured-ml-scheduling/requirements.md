# Requirements Document

## Introduction

This document specifies requirements for a structured machine learning approach to parallel machine scheduling with energy-aware cost optimization. The system transitions from deep learning representation learning to traditional ML models (XGBoost, Gradient Boosting Regressor) operating on manually engineered features from structured/tabular data. The problem involves scheduling jobs on parallel machines with varying energy rates under a repeating price profile, optimizing for total cost while maintaining a meaningful learning component.

## Glossary

- **System**: The structured ML scheduling system
- **Job**: A unit of work with a fixed processing time
- **Machine**: A computational resource with a specific energy rate
- **Price_Profile**: A repeating 20-hour cycle with 3 main price levels
- **Horizon**: The fixed time window for scheduling all jobs
- **Processing_Time**: The duration required to complete a job
- **Energy_Rate**: The power consumption rate of a machine
- **Cost**: The total expense calculated from energy consumption and price profile
- **Feature_Vector**: A structured representation of problem state for ML models
- **DP_Subroutine**: Dynamic programming algorithm that optimally schedules a given sequence on each machine
- **Assignment**: The allocation of jobs to machines
- **Sequence**: The ordering of jobs on a machine
- **ML_Model**: Traditional machine learning model (XGBoost, GBR, etc.)
- **Tabular_Data**: Structured data representation in CSV-like format

## Requirements

### Requirement 1: Feature Engineering for Problem State

**User Story:** As a data scientist, I want to extract meaningful features from the scheduling problem, so that traditional ML models can learn effective scheduling policies.

#### Acceptance Criteria

1. WHEN the system receives problem instance data, THE System SHALL extract features representing the price profile structure
2. WHEN the system receives problem instance data, THE System SHALL extract features representing machine characteristics
3. WHEN the system receives problem instance data, THE System SHALL extract features representing job distribution
4. WHEN the system receives problem instance data, THE System SHALL extract features representing temporal horizon properties
5. THE System SHALL represent all features in tabular format compatible with CSV serialization
6. THE System SHALL normalize or scale features appropriately for ML model consumption

### Requirement 2: Price Profile Feature Extraction

**User Story:** As a data scientist, I want to capture the repeating price pattern structure, so that the ML model can learn cost-aware scheduling decisions.

#### Acceptance Criteria

1. WHEN processing a price profile, THE System SHALL extract the number of complete 20-hour cycles within the horizon
2. WHEN processing a price profile, THE System SHALL identify and encode the 3 main price levels
3. WHEN processing a price profile, THE System SHALL compute statistical features of price distribution (mean, std, min, max)
4. WHEN processing a price profile, THE System SHALL extract the duration of each price level within a cycle
5. WHEN processing a price profile, THE System SHALL compute the position within the current price cycle for any given time point

### Requirement 3: Machine Feature Extraction

**User Story:** As a data scientist, I want to represent machine characteristics as features, so that the ML model can learn energy-aware assignments.

#### Acceptance Criteria

1. WHEN processing machine data, THE System SHALL extract the energy rate for each machine
2. WHEN processing machine data, THE System SHALL compute statistical features of energy rates across all machines (mean, std, min, max)
3. WHEN processing machine data, THE System SHALL compute the relative energy efficiency ranking of each machine
4. WHEN processing machine data, THE System SHALL extract the total number of machines available

### Requirement 4: Job Feature Extraction

**User Story:** As a data scientist, I want to represent job characteristics as features, so that the ML model can learn effective job-to-machine assignments.

#### Acceptance Criteria

1. WHEN processing job data, THE System SHALL extract the distribution of processing times
2. WHEN processing job data, THE System SHALL compute the count of jobs for each unique processing time
3. WHEN processing job data, THE System SHALL compute statistical features of processing times (mean, std, min, max)
4. WHEN processing job data, THE System SHALL compute the total workload (sum of all processing times)
5. WHEN processing job data, THE System SHALL compute the ratio of total workload to available machine capacity

### Requirement 5: Tabular Data Representation

**User Story:** As a data scientist, I want to represent scheduling decisions and states in tabular format, so that I can train traditional ML models.

#### Acceptance Criteria

1. THE System SHALL represent each scheduling decision as a row in tabular format
2. WHEN creating tabular data, THE System SHALL include problem instance features as columns
3. WHEN creating tabular data, THE System SHALL include decision context features as columns
4. WHEN creating tabular data, THE System SHALL include target labels (cost, quality metrics) as columns
5. THE System SHALL serialize tabular data to CSV format for storage and analysis

### Requirement 6: ML Model Training with Structured Data

**User Story:** As a data scientist, I want to train traditional ML models on structured features, so that I can predict high-quality scheduling decisions.

#### Acceptance Criteria

1. THE System SHALL support training XGBoost models on tabular scheduling data
2. THE System SHALL support training Gradient Boosting Regressor models on tabular scheduling data
3. WHEN training ML models, THE System SHALL use extracted features as input
4. WHEN training ML models, THE System SHALL use scheduling quality metrics as targets
5. THE System SHALL persist trained models for inference

### Requirement 7: ML-Guided Job Assignment

**User Story:** As a scheduler, I want the ML model to guide job-to-machine assignments, so that I can achieve low-cost schedules.

#### Acceptance Criteria

1. WHEN assigning jobs to machines, THE System SHALL use the trained ML_Model to predict assignment quality
2. WHEN the ML_Model predicts assignment quality, THE System SHALL consider current machine loads
3. WHEN the ML_Model predicts assignment quality, THE System SHALL consider job processing times
4. WHEN the ML_Model predicts assignment quality, THE System SHALL consider price profile alignment
5. THE System SHALL select assignments based on ML_Model predictions

### Requirement 8: Integration with DP Subroutine

**User Story:** As a scheduler, I want to use the DP subroutine for optimal sequencing, so that I can leverage exact optimization where tractable.

#### Acceptance Criteria

1. WHEN jobs are assigned to machines, THE System SHALL invoke the DP_Subroutine to determine optimal sequencing
2. WHEN the DP_Subroutine completes, THE System SHALL compute the total cost of the resulting schedule
3. THE System SHALL use DP_Subroutine results as ground truth for training data generation

### Requirement 9: Training Data Generation

**User Story:** As a data scientist, I want to generate training data from scheduling instances, so that I can train ML models with meaningful supervision.

#### Acceptance Criteria

1. WHEN generating training data, THE System SHALL create multiple candidate assignments for each problem instance
2. WHEN generating training data, THE System SHALL evaluate each candidate using the DP_Subroutine
3. WHEN generating training data, THE System SHALL label each candidate with its resulting cost
4. WHEN generating training data, THE System SHALL extract features for each candidate assignment
5. THE System SHALL aggregate training data across multiple problem instances

### Requirement 10: Model Evaluation and Validation

**User Story:** As a data scientist, I want to evaluate ML model performance, so that I can verify the learning component is meaningful.

#### Acceptance Criteria

1. WHEN evaluating models, THE System SHALL compute prediction accuracy metrics (MAE, RMSE, R²)
2. WHEN evaluating models, THE System SHALL compare ML-guided schedules against baseline heuristics
3. WHEN evaluating models, THE System SHALL report the cost improvement achieved by ML guidance
4. THE System SHALL validate that the learning component provides statistically significant improvement

### Requirement 11: Feature Importance Analysis

**User Story:** As a data scientist, I want to analyze feature importance, so that I can understand what drives scheduling quality.

#### Acceptance Criteria

1. WHEN analyzing trained models, THE System SHALL extract feature importance scores
2. WHEN analyzing trained models, THE System SHALL rank features by their contribution to predictions
3. THE System SHALL report the top contributing features for scheduling decisions
