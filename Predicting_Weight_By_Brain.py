import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_fwf('brain_body.txt')
x_value = df[['Brain']]
y_value = df[['Body']]
body_reg = linear_model.LinearRegression()
body_reg.fit(x_value, y_value)
plt.scatter(x_value, y_value)
plt.plot(x_value, body_reg.predict(x_value))
plt.show()
