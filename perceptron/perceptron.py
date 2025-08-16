import numpy as np
import seaborn as sns

class Perceptron:

  def __init__(self,epoches=100,learning_rate=0.01):
    self.learning_rate=learning_rate
    self.epoches=epoches
    self.weight=0.0
    self.bias=0.0
    self.history=[]

  def fit(self,x,y):
    x=np.array(x, dtype=float)
    y=np.array(y, dtype=int)
    n_sample,n_feature=x.shape

    # weight initialization
    self.weight=np.zeros(n_feature)

    # bias initialization
    self.bias=0.0

    for _ in range(self.epoches):
      mistake = 0
      for xi,yi in zip(x,y):

        # calculating weighted sum of each sample
        z=np.dot(xi,self.weight) + self.bias

        # linear function 
        y_hat = 1 if z>=0 else 0

        # error calculation 
        error = yi - y_hat

        # updating parameters weights and bias
        if error != 0:
          self.weight += self.learning_rate*error*xi
          self.bias += self.learning_rate*error
          mistake += 1

      self.history.append(mistake)
      if mistake == 0:
        break

  def predict(self, x):
      X = np.array(x, dtype=float)
      z = X @ self.weight + self.bias
      return (z >= 0).astype(int)

  def score(self, x, y):
      y = np.array(y, dtype=int)
      return (self.predict(x) == y).mean()


if __name__ == "__main__":
    
    x = [
    [-1.8371, -2.8679],
    [-1.2572, -0.7838],
    [-2.7372, -3.4215],
    [-0.3208, -1.1316],
    [-2.8332, -2.6306],
    [-3.0679, -3.8369],
    [-2.5688, -5.1508],
    [-4.6049, -4.7097],
    [-3.5400, -3.2306],
    [-3.8286, -5.2396],
    [-0.6954, -2.3915],
    [-2.6609, -4.6692],
    [-3.0266, -3.1935],
    [-3.6910, -3.2398],
    [-3.1908, -3.7104],
    [-2.6775, -1.1383],
    [-2.6700, -4.2774],
    [-1.7059, -3.9715],
    [-2.6197, -5.2263],
    [-3.9466, -3.5607],
    [ 3.3273,  3.6487],
    [ 2.1890,  2.5693],
    [ 0.4530,  1.2491],
    [ 2.1009,  3.9922],
    [ 2.3892,  1.0905],
    [ 2.6965,  2.7324],
    [ 1.7345,  3.3279],
    [ 3.8607,  4.7361],
    [ 1.3187,  2.1254],
    [ 3.0316,  4.3694],
    [ 1.7804,  2.4897],
    [ 0.7853,  0.9008],
    [ 3.7005,  5.1150],
    [ 2.5544,  4.1610],
    [ 2.6791,  2.4428],
    [ 3.2028,  5.0625],
    [ 2.7325,  4.8561],
    [-0.5464,  2.4144],
    [ 2.4327,  2.6934],
    [ 2.0331,  0.6700]
]
y = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

clf = Perceptron(learning_rate=1.0, epoches=5)
clf.fit(x,y)
print("Accuracy:", clf.score(x, y))
print("Weights:", clf.weight, "Bias:", clf.bias)
print("Mistakes per epoch:", clf.history)

    