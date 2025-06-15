# Agentic Financial Fraud Detection System

## Project Overview
This project implements an intelligent fraud detection system using agentic principles. The system processes financial transactions through automated ML pipelines that adapt and improve based on performance metrics.

## Core Features
- **Automated ML Pipeline**: Uses LangGraph for creating sequential learning agents
- **Multiple Model Implementations**: Compares Decision Trees and Random Forests
- **Adaptive Learning**: Implements retraining loops based on performance metrics
- **Comprehensive Evaluation**: Includes metrics tailored for imbalanced fraud detection

## Model Implementations

### 1. Decision Tree Pipeline (agent_dtree.ipynb)
- Basic implementation using sklearn's Decision Tree Classifier
- Features simple hyperparameter tuning
- Good for baseline comparison and interpretability
- Performance metrics focus on basic accuracy and precision

### 2. Random Forest Pipeline (agent_randomforest.ipynb)
- Advanced implementation using Random Forest Classifier
- Includes improver agent for automated retraining
- Features comprehensive hyperparameter optimization
- Implements class balancing for fraud detection
- Contains full retraining loop based on performance thresholds

### 3. Optimized Random Forest (agent_RF_noImprover.ipynb)
- Streamlined version of Random Forest implementation
- Removes retraining loop for production efficiency
- Enhanced cross-validation implementation
- Focused on optimal single-pass performance
- Includes detailed performance metrics visualization


## Technical Architecture
```
📦 Project
├── 📜 agent_dtree.ipynb        # Decision Tree implementation
├── 📜 agent_randomforest.ipynb # Full Random Forest with retraining
├── 📜 agent_RF_noImprover.ipynb# Optimized Random Forest
├── 📜 requirements.txt         # Project dependencies
└── 📜 .gitignore              # Git ignore configurations
```

## Installation and Setup
```bash
git clone https://github.com/saanvi-kanodia/Financial_Fraud_Classifier.git
cd Financial_Fraud_Classifier
pip install -r requirements.txt
```

## Running the Models
1. Place your financial transaction dataset in the project root
2. Open desired notebook (dtree/randomforest/RF_noImprover)
3. Run all cells sequentially
4. View performance metrics and visualizations

## Dependencies
- Python 3.8+
- langgraph>=0.0.15
- scikit-learn>=1.2.0
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

## Project Structure
Each notebook follows an agentic pipeline:
1. **Ingest Agent**: Data loading and preprocessing
2. **EDA Agent**: Exploratory data analysis
3. **Train Agent**: Model training and optimization
4. **Predict Agent**: Fraud prediction
5. **Evaluate Agent**: Performance assessment
6. **Improve Agent**: (In randomforest version) Retraining decisions

## Performance Insights
- Random Forest models show superior performance over Decision Trees
- Removing the improver agent (RF_noImprover) maintains performance while reducing complexity
- All models handle class imbalance effectively through various techniques

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.
