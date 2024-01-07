import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import optuna
import numpy as np
import pickle
import os


DATA_FN = './data/curated_data.csv'
MODEL_PATH = './models'
MODEL_FN = 'xgb.pkl'
TUNING_ITERATIONS = 30
TRAIN_SIZE = 0.7
VISUALIZE_FEATURES = False
LOAD_MODEL = True


def load_data():
    df = pd.read_csv(DATA_FN)
    X = df.drop('account_type', axis=1)
    y = df['account_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0)

    return X_train, X_test, y_train, y_test


def objective(trial, X, y, scoring='accuracy'):
    """ Hyperparameter tuning function for XGBoost
    Args:
        - trial (optuna.trial.Trial): Optuna training trial
        - X (pd.Dataframe): features
        - y (pd.Dataframe): labels

    Returns:
        - float: accuracy
    """
    # Tunable parameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    }

    # 5-fold CV score
    model = XGBClassifier(**params)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    return np.mean(scores)


def train_model(X_train, y_train, n_trials, load_model=False):
    # Load pretrained model
    if load_model:
        model_filepath = os.path.join(MODEL_PATH, MODEL_FN)
        with open(model_filepath, 'rb') as model_file:
            clf = pickle.load(model_file)
            return clf

    # Hyperparameter tuning
    study = optuna.create_study(direction="maximize")
    objective_func = lambda trial: objective(trial, X_train, y_train)
    study.optimize(objective_func, n_trials=n_trials)
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Train classifier with best parameters
    clf = XGBClassifier(**best_params)
    clf.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(MODEL_PATH, MODEL_FN)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    with open(model_path, 'wb') as model_file:
        pickle.dump(clf, model_file)

    return clf


def visualize_feature_importances(X, model, top_n=10):
    feature_importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_dict = dict(zip(feature_names, feature_importances))

    # Sort the features based on their importance
    sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Select the top N features
    top_features = dict(sorted_feature_importances[:top_n])

    # Plot the top features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(top_features.values()), y=list(top_features.keys()), palette='viridis')
    plt.title('Top {} Features'.format(top_n))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('./figures/feature_importances.png')
    plt.show()


if __name__ == '__main__':
    # Tune and train model
    X_train, X_test, y_train, y_test = load_data()
    clf = train_model(X_train, y_train, n_trials=TUNING_ITERATIONS, load_model=LOAD_MODEL)

    # Evaluate on test set
    test_accuracy = clf.score(X_test, y_test)
    print('Test Accuracy:', round(test_accuracy, 4))

    # Visuailze feature importances
    if VISUALIZE_FEATURES:
        visualize_feature_importances(X_train, clf)