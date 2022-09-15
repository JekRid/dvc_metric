import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json

xs = np.linspace(0, 10, 50)
ys = xs**2 + np.random.random(50) * 10
plt.scatter(xs, ys)
plt.show()
model = LinearRegression()
xs = xs.reshape(-1,1)
model.fit(xs, ys)
result = model.predict(xs)
plt.scatter(xs, ys)
plt.scatter(xs, result)
plt.show()
model.score(xs, result)
xs1 = np.c_[xs, xs**2]
model.fit(xs1, ys)
result1 = model.predict(xs1)
plt.scatter(xs, ys)
plt.scatter(xs, result)
plt.scatter(xs, result1)
plt.show()
model.score(xs1, result1)

data = {
"a": (np.sum((ys-result)**2)),
"b": (np.sum((ys-result1)**2))
}

with open('test.json', 'w') as f:
    json.dump(data, f)
