import csv
import os

import numpy
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from prettytable import PrettyTable

from .commons import field_names, table_path, scoring, classifier_save_path
from .model_util import compute_type_detection_rates
from .pickle_util import save_classifier
from conf import SETTINGS

from imblearn.over_sampling import SMOTE

def train_NB_Validate(X_train: numpy.ndarray, y_train: numpy.ndarray, type_labels=None, stratify_labels=None):
   """Train the NB model and validate it using cross validation.
   
   Args:
      X_train: The training set.
      y_train: The labels of the training set.
      type_labels: Malicious type list per sample.
      stratify_labels: Labels used for stratified splitting.
   """
   table = PrettyTable()
   table.field_names = field_names
   csv_path = os.path.join(table_path, "NB_validation.csv")
   k = 4
   skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
   cv_splits = skf
   if stratify_labels is not None:
      label_counts = {}
      for label in stratify_labels:
         label_counts[label] = label_counts.get(label, 0) + 1
      if min(label_counts.values()) >= k:
         cv_splits = list(skf.split(X_train, stratify_labels))
      else:
         print('Warning: Not enough samples per malicious type for stratified CV; falling back to label stratification.')
   smoothings = SETTINGS['classifier']['hyperparameters']['NB']['smoothings']
   type_rates_path = os.path.join(table_path, "NB_validation_types.csv")
   type_rows = []
   
   # 应用SMOTE进行数据平衡
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   
   with open(csv_path, "w+") as f:
      for smoothing in smoothings:
         model = GaussianNB(var_smoothing=smoothing)
         # 使用重采样后的数据进行交叉验证
         scores = cross_validate(model, X=X_resampled, y=y_resampled, cv=skf, scoring=scoring)
         y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=skf)
         tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()
         table.add_row([f"smoothing={smoothing}", tp, fp, tn, fn, scores["test_accu"].mean(), scores["test_prec"].mean(), scores["test_rec"].mean(), scores["test_f1"].mean(), scores["test_matt_cor"].mean()])
         if type_labels is not None:
            y_pred_type = cross_val_predict(model, X_train, y_train, cv=cv_splits)
            for malicious_type, detected, total, rate in compute_type_detection_rates(y_train, y_pred_type, type_labels):
               type_rows.append([f"smoothing={smoothing}", malicious_type, detected, total, rate])
      f.write(table.get_csv_string())
   if type_labels is not None:
      with open(type_rates_path, "w", newline="") as type_file:
         writer = csv.writer(type_file)
         writer.writerow(["hyperparamter", "type", "detected", "total", "rate"])
         writer.writerows(type_rows)

def save_NB(X_train: numpy.ndarray, y_train: numpy.ndarray, smoothing: float):
   """Save the NB model trained on the whole training set.
   
   Args:
      X_train: The training set.
      y_train: The labels of the training set.
      smoothing: The smoothing parameter of the NB.
   """
   save_path = os.path.join(classifier_save_path, "NB.pkl")
   model = GaussianNB(var_smoothing=smoothing)
   model.fit(X_train, y_train)
   save_classifier(model, save_path)
