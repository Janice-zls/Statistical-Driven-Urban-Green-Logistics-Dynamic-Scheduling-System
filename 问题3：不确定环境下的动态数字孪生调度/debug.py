print("Starting...")
try:
    import pandas as pd
    print("pandas imported")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("matplotlib imported")
except Exception as e:
    print("Error:", e)
