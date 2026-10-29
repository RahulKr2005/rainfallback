try:
    from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
    print("Linear models: ALL OK")
except Exception as e:
    print("Linear models Error:", e)

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    print("Metrics: ALL OK")
except Exception as e:
    print("Metrics Error:", e)
