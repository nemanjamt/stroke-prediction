import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(path):
    data = pd.read_csv(path)
    # del data['ever_married']
    # encoding categorical data
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
    X = data.iloc[:, 0:].values

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.7, random_state=0)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.3, random_state=0)

    num = 0
    for y in y_train:
        if y == 1:
            num += 1
    num2 = 0
    for y in y_valid:
        if y == 1:
            num2 += 1
    num3 = 0
    for y in y_test:
        if y == 1:
            num3 += 1
    print(num, num2, num3)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data('healthcare-dataset-stroke-data.csv')

    svc = SVC(kernel='linear', C=2)

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_valid)

    num = 0
    for y in y_pred:
        if y == 1:
            num += 1

    num2 = 0
    for y in y_valid:
        if y == 1:
            num2 += 1
    print("----valid------")
    print("broj predvidjenih kod validacionih ", num)
    print("broj postojecih strokova ", num2)

    accuracy_ = accuracy_score(y_valid, y_pred)
    recall_ = recall_score(y_valid, y_pred, average='macro')
    precision_ = precision_score(y_valid, y_pred, average='macro', zero_division=0)
    f_measure_ = f1_score(y_valid, y_pred, average='macro')
    print("SVM Valid data perfomanse")
    print("Accuracy: ", accuracy_)
    print("Recall: ", recall_)
    print("Precision: ", precision_)
    print("F-measure: ", f_measure_)

    y_pred = svc.predict(X_test)
    num = 0
    for y in y_pred:
        if y == 1:
            num += 1

    num2 = 0
    for y in y_test:
        if y == 1:
            num2 += 1
    print("-----test-----")
    print("broj predvidjenih kod testa svm-a", num)
    print("broj stvarnih kod testa svm-a ", num2)

    print("performanse test svm")
    accuracy_ = accuracy_score(y_test, y_pred)
    recall_ = recall_score(y_test, y_pred, average='macro')
    precision_ = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f_measure_ = f1_score(y_test, y_pred, average='macro')

    print("Accuracy: ", accuracy_)
    print("Recall: ", recall_)
    print("Precision: ", precision_)
    print("F-measure: ", f_measure_)

    print("---RANDOM FOREST-----")


    rf = RandomForestClassifier(n_estimators=1, random_state=168)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)
    accuracy_ = accuracy_score(y_valid, y_pred)
    recall_ = recall_score(y_valid, y_pred, average='macro')
    precision_ = precision_score(y_valid, y_pred, average='macro', zero_division=0)
    f_measure_ = f1_score(y_valid, y_pred, average='macro')
    print("---valid---")
    num = 0
    for y in y_pred:
        if y == 1:
            num += 1

    num2 = 0
    for y in y_valid:
        if y == 1:
            num2 += 1
    print("broj predvidjenih kod validacionih rf-a", num)
    print("broj postojecih strokova rf-a ", num2)
    print("Accuracy: ", accuracy_)
    print("Recall: ", recall_)
    print("Precision: ", precision_)
    print("F-measure: ", f_measure_)

    y_pred = rf.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    recall_ = recall_score(y_test, y_pred, average='macro')
    precision_ = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f_measure_ = f1_score(y_test, y_pred, average='macro')
    print("---test---")
    num = 0
    for y in y_pred:
        if y == 1:
            num += 1

    num2 = 0
    for y in y_test:
        if y == 1:
            num2 += 1
    print("broj predvidjenih kod testa rf-a", num)
    print("broj stvarnih kod testa rf-a ", num2)
    print("Accuracy: ", accuracy_)
    print("Recall: ", recall_)
    print("Precision: ", precision_)
    print("F-measure: ", f_measure_)


if __name__ == '__main__':
    main()
