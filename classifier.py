import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve

import pickle

from data_storing import load_hdf5_train, load_hdf5_test


def preprocess(intervals: pd.Series, window_size: int) -> pd.Series:
    """
    Preprocess the data by applying SMA.
    """
    return intervals.apply(lambda x: x.rolling(window=window_size).mean())

# range is calculated by doing c.max() - c.min()


def feature_extract(intervals):
    return pd.DataFrame.from_records(intervals.apply(lambda c: pd.DataFrame([c.max(), c.min(), c.max() - c.min(), c.mean(), c.median(), c.std(), c.var()]).to_numpy().flatten()))


def classifier_create(save_location: str):
    train_data = load_hdf5_train()

    print("Train data loaded.")

    # Extract features from each interval. Drop NaN valued rows.
    train_data = pd.concat(
        [feature_extract(preprocess(train_data['interval'], 10)), train_data['label']], axis=1).dropna()

    print("Features extracted.")

    # Train the classifier.
    classifier_minmax = make_pipeline(
        MinMaxScaler(), LogisticRegression(random_state=42, max_iter=1000))
    classifier_minmax.fit(train_data.iloc[:, :-1], train_data['label'])

    with open(save_location, 'wb') as f:
        pickle.dump(classifier_minmax, f)

    print("Classifier saved.")


def classifier_test(save_file: str):
    # Prepare the test data.
    test_data = load_hdf5_test()
    test_data = pd.concat(
        [feature_extract(preprocess(test_data['interval'], 10)), test_data['label']], axis=1).dropna()

    # Load the model.
    with open(save_file, 'rb') as f:
        classifier = pickle.load(f)

    # Evaluate the classifier.
    X_test = test_data.iloc[:, :-1]
    Y_test = test_data['label']
    Y_pred = classifier.predict(X_test)
    Y_pred_proba = classifier.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1_score = 2 * (accuracy * recall) / (accuracy + recall)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba[:, 1])
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"ROC AUC: {roc_auc}")

    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_pred), display_labels=[
                           'Walking', 'Jumping']).plot(ax=axis[0])
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=axis[1])
    plt.show()


if __name__ == "__main__":
    classifier_create('classifier.pkl')
    classifier_test('classifier.pkl')
