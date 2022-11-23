import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sb
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def entrena_y_evalua(X_train, X_test, y_train, y_test, sistema):
    if sistema == "KNeighborsClassifier":
        sys = KNeighborsClassifier()
    elif sistema == "GaussianNB":
        sys = GaussianNB()
    elif sistema == "DecisionTreeClassifier":
        sys = DecisionTreeClassifier(random_state=1234)
    elif sistema == "RandomForestClassifier":
        sys = RandomForestClassifier(random_state=1234, n_jobs=-1)
    elif sistema == "SVC":
        sys = SVC(max_iter=100000)
    elif sistema == "LogisticRegression":
        sys = LogisticRegression(max_iter=100000)
    elif sistema == "ovo(rl)":
        sys = OneVsOneClassifier(LogisticRegression(max_iter=100000))
    elif sistema == "ovo(svc)":
        sys = OneVsOneClassifier(SVC(max_iter=1000000))
    elif sistema == "ovr(rl)":
        sys = OneVsRestClassifier(LogisticRegression(max_iter=1000000))
    elif sistema == "ovr(svc)":
        sys = OneVsRestClassifier(SVC(max_iter=1000000))
    else:
        print("Sistema no reconocido")
        exit()

    model = sys.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    resultados = metrics.classification_report(
        y_test, y_pred, output_dict=True)

    return [
        resultados["accuracy"],
        resultados["weighted avg"]["precision"],
        resultados["weighted avg"]["recall"],
        resultados["weighted avg"]["f1-score"],
    ]

df_train = pd.read_csv("train.csv")

le = preprocessing.LabelEncoder()
df_train.drop(["Unnamed: 0"], axis=1, inplace=True)
df_train["lunch"] = le.fit_transform(df_train["lunch"])
df_train["gender"] = le.fit_transform(df_train["gender"])
df_train["test preparation course"] = le.fit_transform(df_train["test preparation course"])
df_train["test preparation course"] = df_train["test preparation course"].replace({0:1, 1:0})
df_train["math score"] /= 100
df_train["reading score"] /= 100
df_train["writing score"] /= 100
df_train["wealthy"] = (df_train["lunch"] + df_train["test preparation course"]) / 2
df_train["score"] = (df_train["math score"] + df_train["reading score"] + df_train["writing score"]) / 3


df_test = pd.read_csv("test.csv")

df_test.drop(["Unnamed: 0"], axis=1, inplace=True)
df_test["lunch"] = le.fit_transform(df_test["lunch"])
df_test["gender"] = le.fit_transform(df_test["gender"])
df_test["test preparation course"] = le.fit_transform(df_test["test preparation course"])
df_test["test preparation course"] = df_test["test preparation course"].replace({0:1, 1:0})
df_test["math score"] /= 100
df_test["reading score"] /= 100
df_test["writing score"] /= 100
df_test["wealthy"] = (df_test["lunch"] + df_test["test preparation course"]) / 2
df_test["score"] = (df_test["math score"] + df_test["reading score"] + df_test["writing score"]) / 3

X = df_train.drop("parental level of education", axis=1)
y = df_train.pop("parental level of education")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)
class_names = ["0", "1", "2", "3", "4", "5"]

sistemas = [
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "ovo(rl)",
    "ovo(svc)",
    "ovr(rl)",
    "ovr(svc)"
]

resultados = np.empty((len(sistemas), 4))
i = 0
for sistema in sistemas:
    resultados[i, :] = entrena_y_evalua(
        X_train, X_test, y_train, y_test, sistema)
    i += 1

df_resultados = pd.DataFrame(
    resultados, index=sistemas, columns=[
        "Accuracy", "Precision", "Recall", "F1-score"]
)
print(df_resultados)


sys = GaussianNB()

model = sys.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, average='macro')

print("Accuracy: %.4f" % acc)
print("F1-Score: %.4f" % f1)

result = model.predict(df_test)
df_result = pd.DataFrame(result, columns=["target"])
out = df_result.to_json("predictions.json", indent=1)