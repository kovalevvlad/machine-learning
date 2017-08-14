from math_util import weekly_periodicity
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

wave1 = pd.DataFrame({"x": np.sin(np.arange(0, 1000) * np.pi * 2 / 7)})
wave2 = 10 * wave1

wave1.plot()
plt.show()

print "periodicity of wave1 = {}".format(weekly_periodicity(wave1, 90))
print "periodicity of wave2 = {}".format(weekly_periodicity(wave2, 90))