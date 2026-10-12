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
