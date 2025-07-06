import numpy as np

def get_data_c(n, min=-273.15, max=1000):
  celsius = np.random.uniform(min, max, size=n)
  data = np.empty((n, 2), dtype=np.float64)
  data[:, 0] = celsius
  data[:, 1] = celsius * 1.8 + 32
  return data

def get_data_f(n, min=-459.67, max=1000):
  fahrenheit = np.random.uniform(min, max, size=n)
  data = np.empty((n, 2), dtype=np.float64)
  data[:, 0] = fahrenheit
  data[:, 1] = (fahrenheit - 32) * 5/9
  return data
