# Decision Tree Classifier - Titanic Survival Prediction

A comprehensive machine learning project demonstrating decision tree classification with data preprocessing, model training, and hyperparameter tuning using both pre-pruning and post-pruning techniques.

## Overview

This project implements a Decision Tree classifier to predict Titanic passenger survival using scikit-learn. The notebook demonstrates complete machine learning workflows including data cleaning, feature engineering, model training, and optimization strategies to prevent overfitting.

## Dataset

- **Source**: Titanic dataset (via Seaborn)
- **Target Variable**: `survived` (0 = Did not survive, 1 = Survived)
- **Features Used**:
  - `pclass`: Passenger class (1st, 2nd, or 3rd)
  - `sex`: Gender (male/female)
  - `fare`: Ticket fare price
  - `embarked`: Port of embarkation
  - `age`: Passenger age

## Project Structure

The notebook follows these main sections:

### 1. **Data Loading & Exploration**
   - Import required libraries (seaborn, pandas, scikit-learn)
   - Load Titanic dataset
   - Examine data structure and missing values

### 2. **Data Preprocessing**
   - **Feature Selection**: Select relevant features for the model
   - **Imputation**: Handle missing values using:
     - Median imputation for `age` column
     - Most frequent value imputation for `embarked` column
   - **Encoding**: Convert categorical features using LabelEncoder:
     - `sex` (male/female → 0/1)
     - `embarked` (port names → encoded values)

### 3. **Train-Test Split**
   - Split data into 70% training and 30% testing (random_state=42)
   - Prepare features (X) and target variable (y)

### 4. **Baseline Decision Tree Model**
   - Train a basic Decision Tree Classifier
   - Calculate accuracy on test set
   - Visualize the tree structure (limited to depth 2 for clarity)

### 5. **Pre-Pruning Optimization**
   - Test multiple `max_depth` values (2-10)
   - Evaluate accuracy for each depth
   - Test `min_samples_split` parameter (5-30 samples)
   - Visualize optimal tree configurations
   - Find best hyperparameters through grid search

### 6. **Post-Pruning Optimization**
   - Train a full unpruned tree
   - Use cost complexity pruning (`ccp_alpha`) technique
   - Generate pruning path with multiple alpha values
   - Train models for each alpha value
   - Identify best alpha that maximizes test set accuracy
   - Visualize the final pruned tree

### 7. **Model Evaluation**
   - Compare accuracy across different approaches
   - Generate tree visualizations with:
     - Class labels (Died/Survived)
     - Feature names
     - Color-coded node information
   - Report best model performance

## Key Techniques

### Pre-Pruning
- Prevents tree growth during training
- Parameters: `max_depth`, `min_samples_split`
- More efficient as it stops early

### Post-Pruning
- Cost complexity pruning (minimal cost-complexity pruning)
- Uses `ccp_alpha` parameter
- Prunes after full tree is grown
- Often produces more optimal results

## Requirements

```
seaborn
matplotlib
pandas
scikit-learn
```

## How to Run

1. Ensure all required packages are installed:
   ```bash
   pip install seaborn matplotlib pandas scikit-learn
   ```

2. Open the notebook in Jupyter or VS Code
3. Run cells sequentially from top to bottom
4. View outputs including:
   - Accuracy scores for different hyperparameters
   - Decision tree visualizations
   - Model comparison results

## Results

The notebook generates:
- **Accuracy metrics** for baseline and optimized models
- **Tree visualizations** showing decision boundaries at different depths
- **Hyperparameter impact analysis** showing how max_depth and min_samples_split affect performance
- **Optimal model** using post-pruning with best ccp_alpha value

## Expected Outcomes

- Baseline accuracy with unpruned tree
- Improved accuracy through hyperparameter tuning
- Insights into the trade-off between model complexity and generalization
- Visualization of how different pruning strategies affect the tree structure

## Key Insights

1. Deeper trees can lead to overfitting
2. Pre-pruning constraints reduce model complexity early
3. Post-pruning (cost complexity pruning) optimizes tree size after full training
4. The optimal tree balances accuracy with model simplicity
5. Hyperparameter tuning significantly impacts model performance

## Technologies Used

- **Python 3**: Programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library
- **matplotlib & seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## License

This project is for educational purposes.