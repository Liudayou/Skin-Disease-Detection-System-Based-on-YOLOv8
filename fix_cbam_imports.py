import glob
for py_file in glob.glob('Experiment/scripts/attention/ablations/*.py'):
    with open(py_file, 'r') as f:
        code = f.read()
    # It currently does:
    """
    try:
        from modules.cbam import register_cbam_for_ultralytics
        register_cbam_for_ultralytics()
    except: pass
    """
    # So the exception is silently caught, meaning it's never registered! 
    code = code.replace("from modules.cbam import register_cbam_for_ultralytics", "from modules.cbam_attention import register_cbam_for_ultralytics")
    with open(py_file, 'w') as f:
        f.write(code)

