
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path)

    data['gender'] = data['gender'].replace(
        {'Male': 0, 'Female': 1, 'Other': 2})

    data['ever_married'] = data['ever_married'].replace(
        {'No': 0, 'Yes': 1})
    data['work_type'] = data['work_type'].replace(
        {'children': 0, 'Govt_job': 1, 'Never_worked': 2, "Private": 3, "Self-employed": 4})
    data['Residence_type'] = data['Residence_type'].replace(
        {'Rural': 0, 'Urban': 1})

    data['smoking_status'] = data['smoking_status'].replace(
        {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2})
    data['smoking_status'] = data['smoking_status'].replace(
        {"Unknown": data['smoking_status'].value_counts().idxmax()})

    data['bmi'].fillna(
        value=data['bmi'].mean(), inplace=True)

    Y = data.iloc[:, 11].values
    del data['stroke']
    del data['id']

    del data['ever_married']
    del data['Residence_type']
    # del data['bmi']
    del data['hypertension']
    del data['heart_disease']

    X = data.iloc[:, 0:].values

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.7, random_state=0)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.3, random_state=0)

    return  X_train, y_train, X_valid, y_valid, X_test, y_test


def data_analysis(path):
    data = pd.read_csv(path)
    pd.set_option("display.max.columns", None)
    data['gender'] = data['gender'].replace(
        {'Male': 0, 'Female': 1, 'Other': 2})

    data['ever_married'] = data['ever_married'].replace(
        {'No': 0, 'Yes': 1})
    data['work_type'] = data['work_type'].replace(
        {'children': 0, 'Govt_job': 1, 'Never_worked': 2, "Private": 3, "Self-employed": 4})
    data['Residence_type'] = data['Residence_type'].replace(
        {'Rural': 0, 'Urban': 1})

    data['smoking_status'] = data['smoking_status'].replace(
        {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2})
    data['smoking_status'] = data['smoking_status'].replace(
        {"Unknown": data['smoking_status'].value_counts().idxmax()})

    data['bmi'].fillna(
        value=data['bmi'].mean(), inplace=True)

    # data.plot(x="bmi", y=["stroke"], kind="pie")
    data.plot.hist(column=["smoking_status"], by="stroke")

    plt.show()


def main():

    # data_analysis('healthcare-dataset-stroke-data.csv')
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data('healthcare-dataset-stroke-data.csv')
    # print("---SVM---")
    # svc = SVC(kernel='rbf')
    #
    # svc.fit(X_train, y_train)
    # y_pred = svc.predict(X_valid)
    #
    #
    # accuracy = accuracy_score(y_valid, y_pred)
    # recall = recall_score(y_valid, y_pred, average='macro')
    # precision = precision_score(y_valid, y_pred, average='macro', zero_division=0)
    # f_measure = f1_score(y_valid, y_pred, average='macro')
    #
    # print("---valid---")
    # print("Accuracy: ", accuracy)
    # print("Recall: ", recall)
    # print("Precision: ", precision)
    # print("F-measure: ", f_measure)

    # y_pred = svc.predict(X_test)
    #
    #
    # print("---test---")
    # accuracy = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred, average='macro')
    # precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    # f_measure = f1_score(y_test, y_pred, average='macro')
    #
    # print("Accuracy: ", accuracy)
    # print("Recall: ", recall)
    # print("Precision: ", precision)
    # print("F-measure: ", f_measure)

    print("---RANDOM FOREST-----")

    rf = RandomForestClassifier(n_estimators=5, random_state=5436)
    rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_valid)
    #
    # accuracy = accuracy_score(y_valid, y_pred)
    # recall = recall_score(y_valid, y_pred, average='macro')
    # precision = precision_score(y_valid, y_pred, average='macro', zero_division=0)
    # f_measure = f1_score(y_valid, y_pred, average='macro')
    # print("---valid---")
    #
    # print("Accuracy: ", accuracy)
    # print("Recall: ", recall)
    # print("Precision: ", precision)
    # print("F-measure: ", f_measure)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f_measure = f1_score(y_test, y_pred, average='macro')
    print("---test---")

    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F-measure: ", f_measure)




if __name__ == '__main__':
    main()

