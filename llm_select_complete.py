"""
LLM-Select: A Prompt-Driven Feature Selection Framework
Complete Implementation with Model Development, Training, Testing, and Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#############################################
# 1. DATA GENERATION AND LOADING
#############################################

def generate_benchmark_datasets():
    """Generate 12 benchmark datasets as described in the paper"""
    datasets = {}
    
    # Dataset specifications from Table 1
    dataset_specs = [
        ('GeneExpress-1K', 54675, 1200, 5, 0.123),
        ('FinanceStock-2K', 28450, 2100, 3, 0.237),
        ('TextCorpus-3K', 89234, 3400, 8, 0.678),
        ('ImageFeatures-1.5K', 16384, 1650, 10, 0.089),
        ('BioMarkers-800', 45123, 850, 4, 0.156),
        ('SocialMedia-4K', 102456, 4200, 6, 0.742),
        ('MedicalImaging-2K', 32768, 2000, 7, 0.114),
        ('CryptoMarket-1.8K', 19876, 1800, 3, 0.285),
        ('ProteinSeq-900', 78912, 950, 6, 0.097),
        ('DocumentClass-5K', 156789, 5100, 12, 0.813),
        ('FaceRecog-2.5K', 25600, 2500, 15, 0.068),
        ('ForexPred-1.2K', 12345, 1250, 2, 0.341)
    ]
    
    for name, n_features, n_samples, n_classes, sparsity in dataset_specs:
        print(f"Generating {name}...")
        
        # Create high-dimensional dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(int(n_features * 0.1), n_classes * 2),
            n_redundant=int(n_features * 0.05),
            n_classes=n_classes,
            n_clusters_per_class=2,
            flip_y=0.01,
            class_sep=0.8,
            random_state=42
        )
        
        # Apply sparsity
        if sparsity > 0:
            sparse_mask = np.random.rand(*X.shape) < sparsity
            X[sparse_mask] = 0
        
        datasets[name] = {'X': X, 'y': y, 'n_features': n_features, 
                         'n_samples': n_samples, 'n_classes': n_classes}
    
    return datasets

#############################################
# 2. LLM-SELECT FRAMEWORK IMPLEMENTATION
#############################################

class LLMSelect:
    """
    LLM-Select Feature Selection Framework
    Implements the complete algorithm from the paper
    """
    
    def __init__(self, n_features_to_select=100, alpha=0.4, beta=0.4, gamma=0.2,
                 temperature=0.7, n_iterations=5, random_state=42):
        """
        Initialize LLM-Select framework
        
        Parameters:
        -----------
        n_features_to_select : int
            Number of features to select (k)
        alpha : float
            Weight for semantic importance (simulated)
        beta : float
            Weight for statistical importance
        gamma : float
            Weight for domain relevance
        temperature : float
            Temperature parameter for score normalization
        n_iterations : int
            Number of refinement iterations
        """
        self.k = n_features_to_select
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_scores_ = None
        
    def compute_statistical_importance(self, X, y):
        """
        Compute traditional statistical importance T(f_i)
        Combines: Mutual Information, Correlation, Chi-square, Variance
        """
        n_features = X.shape[1]
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        
        # Variance
        var_scores = np.var(X, axis=0)
        var_scores = (var_scores - var_scores.min()) / (var_scores.max() - var_scores.min() + 1e-8)
        
        # F-statistic (ANOVA)
        f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, 0)
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        
        # Correlation with target (for binary/ordinal targets)
        corr_scores = np.zeros(n_features)
        for i in range(n_features):
            try:
                corr_scores[i] = abs(pearsonr(X[:, i], y)[0])
            except:
                corr_scores[i] = 0
        corr_scores = np.nan_to_num(corr_scores, 0)
        
        # Weighted combination (Equation from paper)
        w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
        T = w1 * mi_scores + w2 * corr_scores + w3 * f_scores + w4 * var_scores
        
        return T
    
    def compute_semantic_importance(self, X, y):
        """
        Simulate semantic importance S(f_i) using LLM understanding
        In practice, this would use actual LLM embeddings and prompts
        Here we simulate with advanced feature interactions
        """
        n_features = X.shape[1]
        
        # Simulate semantic understanding through feature interaction patterns
        # Use Random Forest feature importance as proxy for semantic relevance
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                    random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        semantic_scores = rf.feature_importances_
        
        # Add noise to simulate LLM variability
        noise = np.random.normal(0, 0.1, n_features)
        semantic_scores = semantic_scores + noise
        semantic_scores = np.clip(semantic_scores, 0, None)
        
        # Normalize
        semantic_scores = (semantic_scores - semantic_scores.min()) / \
                         (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        return semantic_scores
    
    def compute_domain_relevance(self, X):
        """
        Compute domain relevance R(f_i)
        Simulates expert domain knowledge
        """
        n_features = X.shape[1]
        
        # Simulate domain expert ratings
        # Features with higher variance and non-zero values get higher relevance
        non_zero_ratio = np.count_nonzero(X, axis=0) / X.shape[0]
        relevance_scores = non_zero_ratio * np.var(X, axis=0)
        
        # Normalize
        relevance_scores = (relevance_scores - relevance_scores.min()) / \
                          (relevance_scores.max() - relevance_scores.min() + 1e-8)
        
        return relevance_scores
    
    def compute_integrated_importance(self, X, y):
        """
        Compute integrated importance scores I(f_i)
        Equation (1) from paper: I(f_i) = α·S(f_i) + β·T(f_i) + γ·R(f_i)
        """
        print("Computing statistical importance...")
        T = self.compute_statistical_importance(X, y)
        
        print("Computing semantic importance...")
        S = self.compute_semantic_importance(X, y)
        
        print("Computing domain relevance...")
        R = self.compute_domain_relevance(X)
        
        # Integrated importance (Equation 1)
        I = self.alpha * S + self.beta * T + self.gamma * R
        
        return I, S, T, R
    
    def iterative_refinement(self, X, y, I):
        """
        Iterative refinement process (Equation 14)
        I^(t+1)(f_i) = I^(t)(f_i) + η·∇I(f_i) + δ·Feedback(f_i)
        """
        learning_rate = 0.1
        feedback_weight = 0.05
        
        for iteration in range(self.n_iterations):
            # Select top k features
            top_indices = np.argsort(I)[-self.k:]
            
            # Compute feedback based on performance
            X_selected = X[:, top_indices]
            
            # Train quick model for feedback
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, 
                                       random_state=self.random_state)
            scores = cross_val_score(rf, X_selected, y, cv=3, n_jobs=-1)
            performance = scores.mean()
            
            # Update importance scores
            feedback = np.zeros_like(I)
            feedback[top_indices] = performance * 0.1
            
            # Gradient (simplified as difference from mean)
            gradient = I - I.mean()
            
            # Update (Equation 14)
            I = I + learning_rate * gradient + feedback_weight * feedback
            
            print(f"  Iteration {iteration + 1}/{self.n_iterations}: "
                  f"Performance = {performance:.4f}")
        
        return I
    
    def fit(self, X, y):
        """
        Fit the LLM-Select framework
        """
        print(f"\n{'='*60}")
        print(f"LLM-Select Feature Selection")
        print(f"{'='*60}")
        print(f"Dataset shape: {X.shape}")
        print(f"Target classes: {len(np.unique(y))}")
        print(f"Features to select: {self.k}")
        
        start_time = time.time()
        
        # Compute integrated importance scores
        I, S, T, R = self.compute_integrated_importance(X, y)
        
        # Iterative refinement
        print("\nPerforming iterative refinement...")
        I = self.iterative_refinement(X, y, I)
        
        # Select top k features
        self.selected_features_ = np.argsort(I)[-self.k:]
        self.feature_scores_ = I
        self.semantic_scores_ = S
        self.statistical_scores_ = T
        self.relevance_scores_ = R
        
        elapsed_time = time.time() - start_time
        
        print(f"\nFeature selection completed in {elapsed_time:.2f} seconds")
        print(f"Selected {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X):
        """Transform X by selecting features"""
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)

#############################################
# 3. BASELINE METHODS FOR COMPARISON
#############################################

class BaselineSelector:
    """Implements various baseline feature selection methods"""
    
    def __init__(self, method='mutual_info', n_features=100):
        self.method = method
        self.n_features = n_features
        self.selected_features_ = None
        
    def fit(self, X, y):
        if self.method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=42)
        elif self.method == 'f_classif':
            scores, _ = f_classif(X, y)
            scores = np.nan_to_num(scores, 0)
        elif self.method == 'variance':
            scores = np.var(X, axis=0)
        elif self.method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selected_features_ = np.argsort(scores)[-self.n_features:]
        return self
    
    def transform(self, X):
        return X[:, self.selected_features_]

#############################################
# 4. MODEL TRAINING AND EVALUATION
#############################################

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Model"):
    """Train classifier and compute comprehensive metrics"""
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                 random_state=42, n_jobs=-1)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'n_features': X_train.shape[1]
    }
    
    return results, y_pred, clf

#############################################
# 5. COMPREHENSIVE EVALUATION FRAMEWORK
#############################################

def comprehensive_evaluation(datasets, n_features_to_select=100):
    """
    Perform comprehensive evaluation across all datasets
    Compare LLM-Select against baseline methods
    """
    
    methods = {
        'LLM-Select': None,
        'Mutual Information': 'mutual_info',
        'F-Statistic': 'f_classif',
        'Random Forest': 'random_forest',
        'Variance': 'variance'
    }
    
    all_results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n{'='*80}")
        print(f"Evaluating Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        X, y = data['X'], data['y']
        
        # Adjust n_features if dataset is smaller
        k = min(n_features_to_select, X.shape[1] // 10)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Evaluate each method
        for method_name, method_type in methods.items():
            print(f"\n--- {method_name} ---")
            
            try:
                if method_name == 'LLM-Select':
                    # Our proposed method
                    selector = LLMSelect(n_features_to_select=k, random_state=42)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                    X_test_selected = selector.transform(X_test_scaled)
                else:
                    # Baseline methods
                    selector = BaselineSelector(method=method_type, n_features=k)
                    X_train_selected = selector.fit(X_train_scaled, y_train).transform(X_train_scaled)
                    X_test_selected = selector.transform(X_test_scaled)
                
                # Train and evaluate
                results, y_pred, clf = train_and_evaluate(
                    X_train_selected, X_test_selected, y_train, y_test, method_name
                )
                
                results['dataset'] = dataset_name
                results['original_features'] = X.shape[1]
                results['selected_features'] = k
                
                all_results.append(results)
                
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"F1-Score: {results['f1_score']:.4f}")
                print(f"Training Time: {results['train_time']:.2f}s")
                
            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
                continue
    
    return pd.DataFrame(all_results)

#############################################
# 6. VISUALIZATION FUNCTIONS
#############################################

def plot_method_comparison(results_df):
    """Plot comparison of different methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    pivot_acc = results_df.pivot(index='dataset', columns='model', values='accuracy')
    pivot_acc.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('')
    axes[0, 0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # F1-Score comparison
    pivot_f1 = results_df.pivot(index='dataset', columns='model', values='f1_score')
    pivot_f1.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_xlabel('')
    axes[0, 1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Training time comparison
    pivot_time = results_df.pivot(index='dataset', columns='model', values='train_time')
    pivot_time.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_xlabel('')
    axes[1, 0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Average performance across all datasets
    avg_metrics = results_df.groupby('model')[['accuracy', 'f1_score', 'precision', 'recall']].mean()
    avg_metrics.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Average Performance Across All Datasets', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xlabel('')
    axes[1, 1].legend(title='Metric')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('llm_select_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: llm_select_comparison.png")
    plt.show()

def plot_feature_importance_analysis(selector, dataset_name):
    """Plot feature importance analysis for LLM-Select"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top features by integrated importance
    top_k = 30
    top_indices = np.argsort(selector.feature_scores_)[-top_k:]
    top_scores = selector.feature_scores_[top_indices]
    
    axes[0, 0].barh(range(top_k), top_scores)
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_ylabel('Feature Index')
    axes[0, 0].set_title(f'Top {top_k} Features by Integrated Importance')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Score distribution
    axes[0, 1].hist(selector.feature_scores_, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.percentile(selector.feature_scores_, 90), 
                       color='red', linestyle='--', label='90th percentile')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Feature Importance Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Component contribution
    components = ['Semantic', 'Statistical', 'Domain']
    weights = [selector.alpha, selector.beta, selector.gamma]
    axes[1, 0].pie(weights, labels=components, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Importance Score Component Weights')
    
    # Score components for top features
    top_features_data = {
        'Semantic': selector.semantic_scores_[top_indices[-10:]],
        'Statistical': selector.statistical_scores_[top_indices[-10:]],
        'Domain': selector.relevance_scores_[top_indices[-10:]]
    }
    
    x = np.arange(10)
    width = 0.25
    axes[1, 1].bar(x - width, top_features_data['Semantic'], width, label='Semantic')
    axes[1, 1].bar(x, top_features_data['Statistical'], width, label='Statistical')
    axes[1, 1].bar(x + width, top_features_data['Domain'], width, label='Domain')
    axes[1, 1].set_xlabel('Top 10 Features')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Score Components for Top 10 Features')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: feature_importance_{dataset_name}.png")
    plt.show()

#############################################
# 7. MAIN EXECUTION
#############################################

def main():
    """Main execution function"""
    
    print("="*80)
    print("LLM-SELECT: COMPLETE IMPLEMENTATION")
    print("Feature Selection Framework Using Large Language Models")
    print("="*80)
    
    # Generate benchmark datasets
    print("\n[1/4] Generating Benchmark Datasets...")
    datasets = generate_benchmark_datasets()
    print(f"Generated {len(datasets)} benchmark datasets")
    
    # Select subset for demonstration (use all 12 for full evaluation)
    demo_datasets = {k: v for k, v in list(datasets.items())[:3]}
    
    # Comprehensive evaluation
    print("\n[2/4] Running Comprehensive Evaluation...")
    results_df = comprehensive_evaluation(demo_datasets, n_features_to_select=100)
    
    # Display results table
    print("\n[3/4] Results Summary:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('llm_select_results.csv', index=False)
    print("\nSaved: llm_select_results.csv")
    
    # Generate visualizations
    print("\n[4/4] Generating Visualizations...")
    plot_method_comparison(results_df)
    
    # Detailed analysis on one dataset
    print("\nPerforming detailed analysis on sample dataset...")
    sample_data = demo_datasets[list(demo_datasets.keys())[0]]
    X, y = sample_data['X'], sample_data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    selector = LLMSelect(n_features_to_select=100, random_state=42)
    selector.fit(X_train_scaled, y_train)
    
    plot_feature_importance_analysis(selector, list(demo_datasets.keys())[0])
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print("  - llm_select_results.csv")
    print("  - llm_select_comparison.png")
    print(f"  - feature_importance_{list(demo_datasets.keys())[0]}.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    llm_results = results_df[results_df['model'] == 'LLM-Select']
    if not llm_results.empty:
        print(f"\nLLM-Select Performance:")
        print(f"  Average Accuracy: {llm_results['accuracy'].mean():.4f} ± {llm_results['accuracy'].std():.4f}")
        print(f"  Average F1-Score: {llm_results['f1_score'].mean():.4f} ± {llm_results['f1_score'].std():.4f}")
        print(f"  Average Training Time: {llm_results['train_time'].mean():.2f}s")
        
        # Comparison with best baseline
        for dataset in llm_results['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            llm_acc = dataset_results[dataset_results['model'] == 'LLM-Select']['accuracy'].values[0]
            baseline_max = dataset_results[dataset_results['model'] != 'LLM-Select']['accuracy'].max()
            improvement = ((llm_acc - baseline_max) / baseline_max) * 100
            print(f"\n  {dataset}: {improvement:+.2f}% improvement over best baseline")

if __name__ == "__main__":
    main()
