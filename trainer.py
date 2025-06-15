import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Bayesian Hyperparameter Optimization
import optuna
# Suppress Optuna's default logging to keep our logs clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- GPU Acceleration (Conditional Import) ---
# Try to import RAPIDS cuML for GPU-accelerated RandomForest.
# If not available, we'll use the scikit-learn (CPU) version.
#try:
#    import cudf
#    from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
#    GPU_AVAILABLE = True
#    print("✅ cuML found. GPU acceleration is ENABLED.")
#except ImportError:
#    GPU_AVAILABLE = False
#    print("⚠️ cuML not found. Falling back to scikit-learn for CPU-based training.")


# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class FraudModelTrainerAgent:
    """
    A high-performance agent for training a financial fraud detection model.
    It's designed for efficiency, class imbalance handling, and LangGraph state compatibility.
    """

    def __init__(self,
                 n_optuna_trials: int = 10, # Fewer trials needed due to Optuna's intelligence
                 cv_folds: int = 3,         # Fewer folds for faster HPO, 5 is also fine
                 random_state: int = 42):
        self.n_optuna_trials = n_optuna_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        logging.info(f"Trainer Agent initialized. Optuna Trials: {n_optuna_trials}, CV Folds: {cv_folds}")

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduces DataFrame memory usage by downcasting numeric types."""
        logging.info("Optimizing memory usage...")
        mem_before = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.select_dtypes(include=np.number).columns:
            if 'float' in str(df[col].dtype):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif 'int' in str(df[col].dtype):
                df[col] = pd.to_numeric(df[col], downcast='integer')

        mem_after = df.memory_usage(deep=True).sum() / 1024**2
        logging.info(f"Memory reduced from {mem_before:.2f} MB to {mem_after:.2f} MB ({100 * (mem_before - mem_after) / mem_before:.2f}% reduction)")
        return df

    def _create_objective(self, X: pd.DataFrame, y: pd.Series):
        """Creates the objective function for Optuna hyperparameter search."""
        
        def objective(trial: optuna.trial.Trial) -> float:
            # --- Better Parameter Space for Financial Fraud ---
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200, log=True),
                'max_depth': trial.suggest_int('max_depth', 8, 16),
                #'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
                'max_features': trial.suggest_float('max_features', 0.2, 0.8),
            }
            
            
            model = RandomForestClassifier(
                    **params,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=self.random_state
                )

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            # Use F1 macro as it's better for severe imbalance. 'f1' (binary) is also fine.
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
            return np.mean(f1_scores)

        return objective

    def train_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for the agent, compatible with LangGraph.
        This function replaces your original `train_agent` function.
        """
        start_time = time.time()
        logging.info("--- Starting Optimized Model Training Workflow ---")
        
        df = state["df"].copy()
        
        # --- 1. Data Preparation & Optimization ---
        df = self._optimize_memory(df)
        df["isFraud"] = df["isFraud"].astype(int)
        
        features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        target = "isFraud"
        X = df[features]
        y = df[target]

        # Stratified split is crucial and must be done *before* any training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=self.random_state, stratify=y
        )
        
        logging.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
        
        # --- 2. Hyperparameter Optimization with Optuna ---
        logging.info(f"Starting Optuna search for {self.n_optuna_trials} trials (running in parallel)...")
        study = optuna.create_study(direction='maximize')
        objective_func = self._create_objective(X_train, y_train)
        
        # n_jobs=-1 runs trials in parallel, massively speeding up the search
        study.optimize(objective_func, n_trials=self.n_optuna_trials, show_progress_bar=True, n_jobs=-1)
        
        hpo_duration = time.time() - start_time
        logging.info(f"Optuna search finished in {hpo_duration:.2f} seconds.")
        
        best_params = study.best_params
        logging.info(f"Best CV F1 Score from study: {study.best_value:.4f}")
        logging.info(f"Best parameters found: {best_params}")
        
        # --- 3. Final Model Training ---
        logging.info("Training final model on the full training set with best parameters...")

        final_model = RandomForestClassifier(
                **best_params,
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state
            )
        
        final_model.fit(X_train, y_train)
        logging.info("Final model trained.")
        
        # --- 4. Evaluation on Unseen Test Set ---
        logging.info("Evaluating model on the hold-out test set...")
        y_pred = final_model.predict(X_test)
        
        y_test_pd, y_pred_pd = y_test, y_pred

        final_f1_score = f1_score(y_test_pd, y_pred_pd)
        
        print("\n" + "="*50)
        print("          FINAL MODEL PERFORMANCE REPORT")
        print("="*50)
        print(f"Test Set F1 Score: {final_f1_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_pd, y_pred_pd))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_pd, y_pred_pd))
        print("="*50)
        
        total_duration = time.time() - start_time
        logging.info(f"--- Entire training workflow completed in {total_duration:.2f} seconds ---")

        # --- 5. Update LangGraph State ---
        state.update({
            "model": final_model,
            "X_test": X_test,
            "y_test": y_test,
            "best_params": best_params,
            "test_f1_score": final_f1_score,
            "trained": True
        })
        
        return state

# --- Example Usage (How to run this code) ---
if __name__ == '__main__':
    # Create a dummy dataframe that mimics a financial dataset
    # In a real scenario, you would load your data here.
    # E.g., df = pd.read_csv('your_fraud_data.csv')
    print("Generating sample imbalanced data for demonstration...")
    n_samples = 50000
    n_features = 5
    n_fraud = 250 # Create class imbalance
    
    X_normal = np.random.rand(n_samples - n_fraud, n_features)
    X_fraud = np.random.rand(n_fraud, n_features) * 2 + 1 # Make fraud transactions slightly different
    
    X_data = np.vstack([X_normal, X_fraud])
    y_data = np.hstack([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])
    
    # Shuffle the data
    p = np.random.permutation(n_samples)
    X_data, y_data = X_data[p], y_data[p]
    
    sample_df = pd.DataFrame(X_data, columns=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
    sample_df["isFraud"] = y_data.astype(int)

    # This is the state your LangGraph would manage
    initial_state = {
        "df": sample_df,
        "trained": False,
        "model": None,
        "X_test": None,
        "y_test": None
    }

    # Initialize and run the trainer agent
    trainer_agent = FraudModelTrainerAgent(n_optuna_trials=30)
    final_state = trainer_agent.train_model(initial_state)

    print("\n--- Final State ---")
    print(f"Model trained: {final_state['trained']}")
    print(f"Test F1 Score: {final_state['test_f1_score']:.4f}")
    print(f"Model Type: {type(final_state['model'])}")