## Dataset Reference

The dataset used in this project is obtained from the following public repository:

- **Poqemon QoE Dataset**  
  Source: :contentReference[oaicite:0]{index=0}  
  URL: https://github.com/Lamyne/Poqemon-QoE-Dataset  
  Author/Maintainer: Lamyne

This dataset provides Quality of Experience (QoE)–related measurements, including Quality of Data (QoD), Quality of Application (QoA), and Quality of Flow (QoF) features, along with the corresponding Mean Opinion Score (MOS).  
It is used as the primary data source for preprocessing, exploratory analysis, and supervised learning–based MOS prediction in this project.

If you use this project or build upon it, please consider citing or acknowledging the original dataset repository to credit the dataset authors for their work.

## Project Workflow

Before running any analysis scripts, create and activate a virtual environment, then install the packages listed in requirements.txt.

The analysis is conducted in three main stages, you can follow the steps listed below:

1. **Data Preprocessing (`data_preprocessing.py`)**  
   The dataset is inspected for missing values. A missing-value report is generated to compute the missing count and missing rate for each attribute.  
   Results show no missing values, so no imputation is required at this stage.

2. **Data Exploration (`data_exploration.py`)**  
   Exploratory data analysis is performed to uncover patterns and relationships in the dataset.  
   This includes visualizing relationships between QoE influence factors and MOS to identify influential features and guide modeling choices.

3. **MOS Prediction and Model Comparison (`mos_predictor.py`)**  
   In the final stage, supervised learning models are used to predict the Mean Opinion Score (MOS) based on selected QoA, QoD, and QoF features.  
   The categorical feature is one-hot encoded, and all models are evaluated on a held-out test set for fair comparison.

   Multiple models are considered, including Random Forest, Decision Tree, Support Vector Machine, and Neural Network.  
   The comparison highlights the strengths and limitations of different modeling approaches for MOS prediction on tabular QoE data.

4. **The early results (`old_version/old_MOS_predictor.py`)**
   This script contains the initial baseline implementation of MOS prediction used at the early stage of the project. It trains a simple predictive model using a limited set of QoE-related features and outputs basic MOS prediction results.
   
   The file is kept in the `old_version/` directory to preserve the original approach for comparison with later, improved versions. It serves as a reference to illustrate how the modelling pipeline evolved from a straightforward baseline toward a more robust and systematic solution through improved preprocessing, feature handling, and model selection.


This workflow ensures data quality verification precedes exploration, and exploration results directly inform the predictive modeling stage.
