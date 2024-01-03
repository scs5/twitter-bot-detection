import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


DATA_FN = './data/curated_data.csv'


def visualize_feature_importances(model, top_n=10):
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
    df = pd.read_csv(DATA_FN)
    X = df.drop('account_type', axis=1)
    y = df['account_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print('Accuracy:', score)

    visualize_feature_importances(clf)