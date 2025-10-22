# Iris Flower Classification - Complete Machine Learning Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================
print("="*60)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("="*60)

# Load the Iris dataset from CSV file
# Common CSV filenames: iris.csv, Iris.csv, IRIS.csv
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('Iris.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('IRIS.csv')
        except FileNotFoundError:
            print("Error: Could not find iris.csv file in the current directory.")
            print("Please ensure the CSV file is in the same folder as this script.")
            exit()

print("\n1. DATASET OVERVIEW")
print("-" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {len(df)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nDataset Statistics:")
print(df.describe())

# ============================================
# 2. DATA PREPROCESSING
# ============================================
print("\n2. DATA PREPROCESSING")
print("-" * 60)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle common column name variations
# Try to identify feature and target columns
possible_target_names = ['species', 'Species', 'variety', 'Variety', 'class', 'Class']
target_column = None

for col in possible_target_names:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # Assume last column is target
    target_column = df.columns[-1]
    print(f"\nAssuming '{target_column}' is the target column")

# Separate features and target
X = df.drop(columns=[target_column])

# Remove 'Id' or 'id' column if present
id_columns = [col for col in X.columns if col.lower() in ['id', 'index']]
if id_columns:
    X = X.drop(columns=id_columns)
    print(f"Removed ID column: {id_columns}")

y = df[target_column]

# Encode target labels if they are strings
if y.dtype == 'object':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    species_names = le.classes_
    print(f"\nSpecies found: {list(species_names)}")
    print(f"Encoded as: {dict(zip(species_names, range(len(species_names))))}")
else:
    y_encoded = y.values
    species_names = [f"Class {i}" for i in np.unique(y_encoded)]

print(f"\nFeatures used: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

print("\nClass Distribution:")
print(pd.Series(y).value_counts())

# ============================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================
print("\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 60)

# Correlation matrix
print("\nFeature Correlation Matrix:")
correlation = X.corr()
print(correlation)

# ============================================
# 4. DATA PREPARATION
# ============================================
print("\n4. DATA PREPARATION")
print("-" * 60)

# Convert features to numpy array
X_array = X.values

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Feature scaling (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 5. TRAIN MULTIPLE MODELS
# ============================================
print("\n5. MODEL TRAINING AND EVALUATION")
print("-" * 60)

# Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Use scaled data for models that benefit from it
    if name in ['Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=species_names))

# ============================================
# 6. MODEL COMPARISON
# ============================================
print("\n6. MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':<15}")
print("-" * 60)

for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<30} {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with accuracy of {results[best_model]:.4f}")

# ============================================
# 7. KEY INSIGHTS
# ============================================
print("\n7. KEY INSIGHTS")
print("=" * 60)
print("""
Classification Concepts Demonstrated:
- Supervised Learning: We trained models using labeled data
- Train-Test Split: 80% training, 20% testing to prevent overfitting
- Feature Scaling: Standardization improves performance of distance-based models
- Cross-Validation: 5-fold CV ensures model reliability
- Multiple Algorithms: Compared different approaches to find the best one
- Performance Metrics: Accuracy, precision, recall, and F1-score

The Iris dataset is linearly separable for most species, which explains
the high accuracy rates. Setosa is easily distinguishable, while
Versicolor and Virginica have some overlap in feature space.
""")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)