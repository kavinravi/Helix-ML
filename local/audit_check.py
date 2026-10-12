from audit import check_source

task = {"policy": dict.fromkeys(["pretrained", "ensemble", "features", "tuning", "augmentation", "regularization"], False)}
check_source("model.to(device).eval()", task)
check_source("model.eval()", task)
for code in ["eval(code)", "exec(code)", "builtins.eval(code)", "frame.eval('x+y')", "model.eval(code='x')", "model.eval(*args)", "from builtins import eval as run; run(code)"]:
    try:
        check_source(code, task)
    except ValueError as error:
        assert "Dynamic code" in str(error)
    else:
        raise AssertionError(f"Accepted dynamic evaluation: {code}")
print("Model inference mode allowed; dynamic string evaluation still rejected.")
cluster = {"policy": task["policy"], "learning": "clustering", "model": "cuml.cluster.KMeans"}
check_source("from cuml.cluster import KMeans; KMeans(n_clusters=2)", cluster)
check_source("from cuml.preprocessing import StandardScaler; StandardScaler()", cluster)
for code in ["from cuml.cluster import DBSCAN; DBSCAN()", "import cuml; cuml.DBSCAN()", "from sklearn.cluster import DBSCAN; DBSCAN()"]:
    try:
        check_source(code, cluster)
    except ValueError as error:
        assert "requested estimator KMeans" in str(error)
    else:
        raise AssertionError(f"Accepted wrong cuML model: {code}")
print("cuML model restrictions are checked across GPU and CPU estimator imports.")
