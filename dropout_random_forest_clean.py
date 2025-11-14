import os
import shutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.utils import resample

# -------------------------
# Utility / modular functions
# -------------------------

def handle_missing_values(df):
    """
    Handle missing values in the dataset:
    - Fill numeric columns with median.
    - Fill categorical columns with mode (most frequent value).
    - Report missing stats for transparency.
    - Save missing values information to CSV.
    """
    df_work = df.copy()
    missing_stats = {}
    missing_records = []

    for col in df_work.columns:
        n_missing = df_work[col].isnull().sum()
        if n_missing > 0:
            if np.issubdtype(df_work[col].dtype, np.number):
                fill_value = df_work[col].median()
                data_type = 'numeric'
            else:
                # If mode is empty (e.g. all NaN), assign "Unknown"
                if df_work[col].mode().empty:
                    fill_value = "Unknown"
                else:
                    fill_value = df_work[col].mode()[0]
                data_type = 'categorical'
            
            df_work[col].fillna(fill_value, inplace=True)
            missing_stats[col] = {
                'missing_count': int(n_missing),
                'fill_value_used': fill_value
            }
            
            # Add record for CSV
            missing_records.append({
                'column_name': col,
                'data_type': data_type,
                'missing_count': int(n_missing),
                'missing_percentage': float((n_missing / len(df)) * 100),
                'fill_value_used': fill_value
            })
    
    # Save missing values information to CSV
    if missing_records:
        missing_df = pd.DataFrame(missing_records)
        missing_df = missing_df.sort_values('missing_count', ascending=False)
        missing_df.to_csv('missing_values_report.csv', index=False)
        # FIXED: Replace Unicode arrow with ASCII
        print(f"  Missing values report saved -> missing_values_report.csv")
    
    print("  Missing values filled using median (numeric) and mode (categorical).")
    return df_work, missing_stats


def cap_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Cap outliers using IQR method (winsorization) instead of removing them
    """
    df_work = df.copy()
    capping_stats = {}
    
    if columns is None:
        columns = df_work.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_work.columns or not np.issubdtype(df_work[col].dtype, np.number):
            continue

        Q1 = df_work[col].quantile(0.25)
        Q3 = df_work[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
            
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        lower_outliers = (df_work[col] < lower).sum()
        upper_outliers = (df_work[col] > upper).sum()
        total_outliers = lower_outliers + upper_outliers

        df_work[col] = np.where(df_work[col] < lower, lower, 
                       np.where(df_work[col] > upper, upper, df_work[col]))

        capping_stats[col] = {
            'Q1': float(Q1), 'Q3': float(Q3), 'IQR': float(IQR),
            'lower_bound': float(lower), 'upper_bound': float(upper),
            'lower_outliers': int(lower_outliers),
            'upper_outliers': int(upper_outliers),
            'total_outliers': int(total_outliers),
            'outliers_pct': float((total_outliers / len(df_work)) * 100)
        }

    # Only show completion message if outliers were actually capped
    if capping_stats and any(stats['total_outliers'] > 0 for stats in capping_stats.values()):
        # FIXED: Remove Unicode characters
        print("  Outlier capping completed successfully.\n")
    
    return df_work, capping_stats


def select_features_rf(X, y, importance_threshold=0.01, random_state=42, n_estimators=200):
    if hasattr(y, 'dtype') and y.dtype == 'object':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = y

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y_enc)
    imp = rf.feature_importances_

    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': imp}).sort_values('Importance', ascending=False)
    selected_features = feat_imp.loc[feat_imp['Importance'] > importance_threshold, 'Feature'].tolist()

    stats = {
        'original_features': X.shape[1],
        'selected_features': len(selected_features),
        'discarded_features': X.shape[1] - len(selected_features),
        'importance_threshold': importance_threshold,
        'selection_ratio': len(selected_features) / X.shape[1]
    }
    return selected_features, feat_imp, stats


def preprocess_data(df, target_column, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_enc = pd.get_dummies(X, drop_first=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    preprocessor = {
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': X_enc.columns.tolist()
    }

    return X_train, X_test, y_train, y_test, preprocessor, X_enc


def perform_bootstrapping(model, X_test, y_test, preprocessor, n_bootstrap=1000, random_state=42):
    print(f"\n  Performing bootstrapping with {n_bootstrap} iterations...")
    
    np.random.seed(random_state)
    bootstrap_scores = []
    n_samples = len(y_test)
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_test[indices]
        y_bootstrap = y_test[indices]
        y_pred = model.predict(X_bootstrap)
        f1 = f1_score(y_bootstrap, y_pred, average='weighted')
        bootstrap_scores.append(f1)
    
    alpha = 0.95
    lower_percentile = (1 - alpha) / 2 * 100
    upper_percentile = (alpha + (1 - alpha) / 2) * 100
    
    lower_ci = np.percentile(bootstrap_scores, lower_percentile)
    upper_ci = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    
    bootstrap_results = {
        'bootstrap_scores': bootstrap_scores,
        'mean_f1_weighted': float(mean_score),
        'confidence_interval_95': (float(lower_ci), float(upper_ci)),
        'bootstrap_std': float(np.std(bootstrap_scores)),
        'n_bootstrap': n_bootstrap
    }
    
    print(f"  Bootstrap Results (F1-weighted):")
    print(f"    Mean: {mean_score:.4f}")
    print(f"    95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
    print(f"    Std: {np.std(bootstrap_scores):.4f}")
    
    return bootstrap_results


def perform_descriptive_statistics(df):
    print("\n--- Descriptive Statistics for Numeric Variables ---")
    desc_stats = df.describe().loc[['mean', '50%', 'std', 'min', 'max']]
    desc_stats.rename(index={'50%': 'median'}, inplace=True)
    print(desc_stats)


def display_metrics_with_formulas(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nMetrics per class (manual computation with formulas):")
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        accuracy = (TP + TN) / cm.sum()
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        print(f"\nClass '{cls}':")
        print(f"  Accuracy  = (TP + TN) / (TP + TN + FP + FN) = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = {accuracy:.3f}")
        print(f"  Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.3f}")
        print(f"  Recall    = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.3f}")
        print(f"  F1 Score  = 2 * (Precision * Recall) / (Precision + Recall) = {f1:.3f}")

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("\nWeighted averages (multi-class):")
    print(f"  Accuracy  = {accuracy:.3f}")
    print(f"  Precision = {precision:.3f}")
    print(f"  Recall    = {recall:.3f}")
    print(f"  F1 Score  = {f1:.3f}")


def train_evaluate_rf(X_train, X_test, y_train, y_test, preprocessor=None, param_grid=None, cv_folds=5, random_state=42, bootstrap=True):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid=param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    class_names = preprocessor['label_encoder'].classes_ if preprocessor else None
    if class_names is not None:
        display_metrics_with_formulas(y_test, y_pred, class_names)

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    print("CV Scores (f1_macro):", ['{:.3f}'.format(score) for score in cv_scores])
    print("Average CV Score:", '{:.3f}'.format(np.mean(cv_scores)))

    bootstrap_results = perform_bootstrapping(model, X_test, y_test, preprocessor) if bootstrap else None

    results = {
        'best_params': grid.best_params_,
        'best_cv_score_f1_macro': float(grid.best_score_),
        'holdout_accuracy': float(accuracy_score(y_test, y_pred)),
        'holdout_f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
        'cv_f1_macro_mean': float(np.mean(cv_scores)),
        'cv_f1_macro_std': float(np.std(cv_scores)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'bootstrap_results': bootstrap_results
    }
    return model, grid, results


def create_model_visualizations(model, X_test, y_test, preprocessor, feature_names, bootstrap_results=None, save_path='.'):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('model', exist_ok=True)  # Ensure model directory exists
    
    le = preprocessor['label_encoder']

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        plt.figure(figsize=(10, max(6, len(fi)//4)))
        plt.barh(fi['Feature'], fi['Importance'])
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        # Save in both locations
        plt.savefig(os.path.join(save_path, 'rf_feature_importance.png'), dpi=150)
        plt.savefig(os.path.join('model', 'rf_feature_importance.png'), dpi=150)
        plt.close()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # Save in both locations
    plt.savefig(os.path.join(save_path, 'confusion_matrix_test.png'), dpi=150)
    plt.savefig(os.path.join('model', 'confusion_matrix_test.png'), dpi=150)
    plt.close()

    # Class Distribution
    y_labels = le.inverse_transform(y_test)
    pd.Series(y_labels).value_counts().plot(kind='bar', color='teal')
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    # Save in both locations
    plt.savefig(os.path.join(save_path, 'class_distribution.png'), dpi=150)
    plt.savefig(os.path.join('model', 'class_distribution.png'), dpi=150)
    plt.close()

    # Bootstrap Distribution
    if bootstrap_results is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(bootstrap_results['bootstrap_scores'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(bootstrap_results['mean_f1_weighted'], color='red', linestyle='--', linewidth=2, label=f"Mean: {bootstrap_results['mean_f1_weighted']:.4f}")
        plt.axvline(bootstrap_results['confidence_interval_95'][0], color='orange', linestyle='--', linewidth=1, label=f"95% CI: [{bootstrap_results['confidence_interval_95'][0]:.4f}, {bootstrap_results['confidence_interval_95'][1]:.4f}]")
        plt.axvline(bootstrap_results['confidence_interval_95'][1], color='orange', linestyle='--', linewidth=1)
        plt.xlabel('F1 Score (Weighted)')
        plt.ylabel('Frequency')
        plt.title('Bootstrap Distribution of F1 Scores')
        plt.legend()
        plt.tight_layout()
        # Save in both locations
        plt.savefig(os.path.join(save_path, 'bootstrap_distribution.png'), dpi=150)
        plt.savefig(os.path.join('model', 'bootstrap_distribution.png'), dpi=150)
        plt.close()
        # FIXED: Replace Unicode arrow with ASCII
        print("  Bootstrap distribution plot saved -> bootstrap_distribution.png")

def save_model_artifacts(model, preprocessor, results, feature_names, selected_features, dropped_features, model_dir='model'):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    print("Saving model artifacts in directory:", model_dir)

    joblib.dump(model, os.path.join(model_dir, 'rf_model.pkl'))
    joblib.dump(preprocessor['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(preprocessor['label_encoder'], os.path.join(model_dir, 'encoder.pkl'))
    joblib.dump(results, os.path.join(model_dir, 'evaluation_results.pkl'))

    with open(os.path.join(model_dir, 'features.txt'), 'w', encoding='utf-8') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")

    pd.DataFrame({'Selected_Features': selected_features}).to_csv(os.path.join(model_dir, 'selected_features.csv'), index=False)
    pd.DataFrame({'Dropped_Features': dropped_features}).to_csv(os.path.join(model_dir, 'dropped_features.csv'), index=False)

    # FIXED: Remove Unicode characters
    print(f"Model and artifacts saved in '{model_dir}/' directory.")
    return model_dir


# -------------------------
# MAIN PIPELINE
# -------------------------
def main(handle_outliers=True, outlier_multiplier=1.5, bootstrap=True):
    # FIXED: Remove Unicode characters from headers
    print("="*60)
    print("BIAI 3110 - RANDOM FOREST PIPELINE (WITH BOOTSTRAPPING)")
    print("="*60)

    print("\nTASK 1: Loading dataset")
    if not os.path.exists('data.csv'):
        raise FileNotFoundError("Place 'data.csv' in the working directory.")
    df = pd.read_csv('data.csv')
    print(f"  Dataset shape: {df.shape}")

    print("\nTASK 2: Missing Values Check")
    if df.isnull().sum().sum() == 0:
        print("  No missing values detected (dataset clean).")
        # Create empty missing values report
        empty_missing_df = pd.DataFrame(columns=['column_name', 'data_type', 'missing_count', 'missing_percentage', 'fill_value_used'])
        empty_missing_df.to_csv('missing_values_report.csv', index=False)
        # FIXED: Replace Unicode arrow with ASCII
        print("  Empty missing values report saved -> missing_values_report.csv")
    else:
        print("  Missing values found - filling missing data.")
        df, missing_stats = handle_missing_values(df)
        print(f"  Missing values filled. Dataset shape: {df.shape}")

    if handle_outliers:
        print(f"\nTASK 3: Detecting and handling outliers (IQR Method, multiplier={outlier_multiplier})")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_candidates = [c for c in df.columns if c.lower() in ('target', 'status', 'outcome')]
        if target_candidates and target_candidates[0] in numeric_cols:
            numeric_cols.remove(target_candidates[0])
        df, stats = cap_outliers_iqr(df, numeric_cols, multiplier=outlier_multiplier)
        if stats and any(col_stats['total_outliers'] > 0 for col_stats in stats.values()):
            print(f"\n  Outlier capping summary:")
            total_outliers = 0
            columns_with_outliers = {col: col_stats for col, col_stats in stats.items() if col_stats['total_outliers'] > 0}
            for col, col_stats in list(columns_with_outliers.items())[:8]:
                print(f"    {col}: {col_stats['total_outliers']} outliers capped ({col_stats['outliers_pct']:.1f}%)")
                total_outliers += col_stats['total_outliers']
            if len(columns_with_outliers) > 8:
                print(f"    ... and {len(columns_with_outliers) - 8} more columns had outliers")
            print(f"  Total outliers capped: {total_outliers}")
        else:
            print("  No outliers detected in the dataset.")
        print(f"  Dataset shape remains: {df.shape}")
    else:
        print("\nTASK 3: Outlier handling skipped.")

    print("\nTASK 4: Detect Target Column")
    candidates = [c for c in df.columns if c.lower() in ('target', 'status', 'outcome')]
    target_col = candidates[0] if candidates else None
    if not target_col:
        raise ValueError("Target column not found.")
    print(f"  Target column: {target_col}")
    print(df[target_col].value_counts())

    perform_descriptive_statistics(df)

    print("\nTASK 5: Feature Selection (Random Forest)")
    X_temp = df.drop(columns=[target_col])
    y_temp = df[target_col]
    X_enc = pd.get_dummies(X_temp, drop_first=True)

    selected, fi, stats = select_features_rf(X_enc, y_temp, importance_threshold=0.01)
    dropped = [feat for feat in X_enc.columns if feat not in selected]

    print(f"\n  Selected features ({len(selected)}):")
    for feat in selected:
        print(f"    - {feat}")

    print(f"\n  Dropped features ({len(dropped)}):")
    for feat in dropped:
        print(f"    - {feat}")

    print("\n  Top 10 important features:\n", fi.head(10).to_string(index=False))

    clean_df = pd.concat([df[[target_col]].reset_index(drop=True),
                          X_enc[selected].reset_index(drop=True)], axis=1)
    clean_df.to_csv('clean_student_data.csv', index=False)
    # FIXED: Replace Unicode arrow with ASCII
    print("  Saved cleaned dataset -> clean_student_data.csv")

    print("\nTASK 6: Data Preprocessing & 70/30 Split")
    X_train, X_test, y_train, y_test, preprocessor, X_encoded = preprocess_data(clean_df, target_col)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nTASK 7: Model Training & Evaluation")
    best_model, grid, results = train_evaluate_rf(X_train, X_test, y_train, y_test, 
                                                 preprocessor=preprocessor, bootstrap=bootstrap)
    print("  Best Params:", results['best_params'])

    # FIXED: Remove Unicode characters from performance headers
    print("\nMODEL PERFORMANCE")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    print("CV Score (f1_macro): {:.3f}".format(np.mean(cv_scores)))
    print("CV Scores for each fold:", ['{:.3f}'.format(score) for score in cv_scores])

    if bootstrap and results.get('bootstrap_results'):
        bs = results['bootstrap_results']
        # FIXED: Remove Unicode characters
        print(f"\nBOOTSTRAP RESULTS (F1-weighted, {bs['n_bootstrap']} iterations):")
        print(f"   Mean F1 Score: {bs['mean_f1_weighted']:.4f}")
        print(f"   95% Confidence Interval: [{bs['confidence_interval_95'][0]:.3f}, {bs['confidence_interval_95'][1]:.3f}]")
        print(f"   Standard Deviation: {bs['bootstrap_std']:.4f}")
        print(f"   Range: {np.min(bs['bootstrap_scores']):.4f} - {np.max(bs['bootstrap_scores']):.4f}")

    print("\nTASK 8: Creating Visualizations")
    create_model_visualizations(best_model, X_test, y_test, preprocessor, 
                              preprocessor['feature_names'], results.get('bootstrap_results'))
    # FIXED: Replace Unicode arrow with ASCII
    print("  Visuals saved: rf_feature_importance.png, confusion_matrix_test.png, class_distribution.png" + 
          (", bootstrap_distribution.png" if bootstrap else ""))

    print("\nTASK 9: Saving Model & Artifacts")
    save_model_artifacts(best_model, preprocessor, results, preprocessor['feature_names'], selected, dropped)

    print("\nTASK 10: Final Summary")
    print(f"  Final dataset shape: {clean_df.shape}")
    print(f"  Features kept: {len(selected)}")
    if bootstrap and results.get('bootstrap_results'):
        bs = results['bootstrap_results']
        print(f"  Bootstrap 95% CI for F1: [{bs['confidence_interval_95'][0]:.3f}, {bs['confidence_interval_95'][1]:.3f}]")
    # FIXED: Remove Unicode characters
    print("  Pipeline execution completed successfully [OK]")

    return best_model, preprocessor, results, clean_df


if __name__ == "__main__":
    main(handle_outliers=True, outlier_multiplier=1.5, bootstrap=True)