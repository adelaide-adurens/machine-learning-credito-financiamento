import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score

from sklearn.metrics import precision_recall_curve

###########################

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from hyperopt import hp, tpe, fmin, Trials, space_eval


###########################
    
def metrica_classificacao(estimator, X_train, X_test, y_train, y_test):
    
    print("Coeficientes:\n", estimator.coef_)
    print("\nIntercept:", estimator.intercept_)

    # dê uma olhada nas classes do modelo
    classes =  estimator.classes_
    print("\nClasses:", classes)

    # probabilidades das previsões treino
    probs_train = estimator.predict_proba(X_train)

    # probabilidade de pertencimento à classe 1 treino
    probs_1_train = probs_train[:, 1]
    
    #calculado a ROC treino
    fpr, tpr, thresholds = roc_curve(y_train, probs_1_train)
    plt.title("TRAIN ROC curve")

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    RocCurveDisplay.from_predictions(y_train, probs_1_train)

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    roc_score_train = roc_auc_score(y_train, probs_1_train )
    
    print("\nROC AUC SCORE TREINO:", roc_score_train)
    
    # probabilidades das previsões de teste
    probs = estimator.predict_proba(X_test)

    # probabilidade de pertencimento à classe 1 de teste
    probs_1 = probs[:, 1]
    
    #calculado a ROC de teste
    fpr, tpr, thresholds = roc_curve(y_test, probs_1)
    plt.title(" TEST ROC curve")

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    RocCurveDisplay.from_predictions(y_test, probs_1)

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    roc_score = roc_auc_score(y_test, probs_1 )
    
    print("\nROC AUC SCORE TESTE:", roc_score)

def metrica_classificacao_pipe(estimator, X_train, X_test, y_train, y_test):
    
    # probabilidades das previsões treino
    probs_train = estimator.predict_proba(X_train)

    # probabilidade de pertencimento à classe 1 treino
    probs_1_train = probs_train[:, 1]
    
    #calculado a ROC treino
    fpr, tpr, thresholds = roc_curve(y_train, probs_1_train)
    plt.title("TRAIN ROC curve")

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    RocCurveDisplay.from_predictions(y_train, probs_1_train)

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    roc_score_train = roc_auc_score(y_train, probs_1_train )
    
    print("\nROC AUC SCORE TREINO:", roc_score_train)
    
    # probabilidades das previsões de teste
    probs = estimator.predict_proba(X_test)

    # probabilidade de pertencimento à classe 1 de teste
    probs_1 = probs[:, 1]
    
    #calculado a ROC de teste
    fpr, tpr, thresholds = roc_curve(y_test, probs_1)
    plt.title("TEST ROC curve")

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    RocCurveDisplay.from_predictions(y_test, probs_1)

    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, ls=":", color="black")

    plt.show()
    
    roc_score = roc_auc_score(y_test, probs_1 )
    
    print("\nROC AUC SCORE TESTE:", roc_score)

    
def pipe_pre_process(df, drop_not_features=["TARGET"], col_target="TARGET",
                             inputer_num_strategy="mean",
                             encoding="onehot"):
    '''
    - inputer_num_strategy (str): "mean", "median";
    - encoding (str): "onehot" para OneHotEncoder; "ordinal" OrdinalEncoder;
    '''
    

    X = df.drop(columns=drop_not_features)
    y = df[col_target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================================

    pipe_features_num = Pipeline([("input_num", SimpleImputer(strategy=inputer_num_strategy)),
                                  ("std", StandardScaler())])

    features_num = X_train.select_dtypes(include=np.number).columns.tolist()

    # ==========================================================

    if encoding == "onehot":
    
        pipe_features_cat = Pipeline([("input_cat", SimpleImputer(strategy="constant", fill_value="unknown")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        
    elif encoding == "ordinal":
        
        pipe_features_cat = Pipeline([("input_cat", SimpleImputer(strategy="constant", fill_value="unknown")),
                                      ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                                 unknown_value=-1))])
    
    else:
        
        raise ValueError("Únicos encodings disponíveis são 'ordinal' e 'onehot'")
        
        
    features_cat = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # ==========================================================

    pre_processador = ColumnTransformer([("transf_num", pipe_features_num, features_num),
                                         ("transf_cat", pipe_features_cat, features_cat)])

    return X_train, X_test, y_train, y_test, pre_processador