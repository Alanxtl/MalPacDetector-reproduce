from sklearn.metrics import (
   accuracy_score,
   classification_report,
   confusion_matrix,
   f1_score,
   matthews_corrcoef,
   precision_score,
   recall_score,
)


def evaluate_model(y_test, y_pred):
   """Evaluate the model.

   Args:
      y_test (list): List of true labels.
      y_pred (list): List of predicted labels.
   """
   tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
   acc = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred, pos_label="malicious")
   recall = recall_score(y_test, y_pred, pos_label="malicious")
   f1 = f1_score(y_test, y_pred, pos_label="malicious")
   mcc = matthews_corrcoef(y_test, y_pred)
   return [tp, fp, tn, fn, acc, precision, recall, f1, mcc]


def build_classification_report(y_test, y_pred):
   """Return a sklearn-style classification report string."""
   return classification_report(
      y_test,
      y_pred,
      labels=["benign", "malicious"],
      target_names=["benign", "malicious"],
      digits=2,
      zero_division=0,
   )


def compute_type_detection_rates(y_true, y_pred, type_labels):
   """Compute detection rates by malicious type.

   Args:
      y_true (list): List of true labels.
      y_pred (list): List of predicted labels.
      type_labels (list[list[str]]): Malicious type list per sample.
   """
   type_totals = {}
   type_detected = {}
   for true_label, pred_label, types in zip(y_true, y_pred, type_labels):
      if true_label != "malicious":
         continue
      if not types:
         continue
      for malicious_type in types:
         type_totals[malicious_type] = type_totals.get(malicious_type, 0) + 1
         if pred_label == "malicious":
            type_detected[malicious_type] = type_detected.get(malicious_type, 0) + 1
   results = []
   for malicious_type in sorted(type_totals.keys()):
      total = type_totals[malicious_type]
      detected = type_detected.get(malicious_type, 0)
      rate = detected / total if total else 0.0
      results.append([malicious_type, detected, total, rate])
   return results
