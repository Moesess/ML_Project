import numpy as np

class LogisticRegression():

  def __init__(self,learning_rate=0.001, number_of_iterations = 1000):
    self.learning_rate = learning_rate
    self.number_of_iterations = number_of_iterations
    self.weights = None
    self.bias = None

  @staticmethod
  def Sigmoidfunction(x):
    return 1/(1+np.exp(-x))

  def fit(self, X, y):
    number_of_samples, number_of_features = X.shape
    self.weights = np.zeros(number_of_features)
    self.bias = 0

    for i in range(self.number_of_iterations):
      linear_predictions = np.dot(X, self.weights) + self.bias
      y_predictions = self.Sigmoidfunction(linear_predictions)

      #print(f"y_predictions: {y_predictions}")
      #print(f"y: {y}")

      dw =(1/number_of_samples)* np.dot(X.transpose(),(y_predictions - y))
      db = (1/number_of_samples)* np.sum(y_predictions - y)
      #print(f"dw: {dw}")
      #print(f"db: {db}")
      self.weights = self.weights - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db


  def predict(self, X):
    linear_predictions = np.dot(X, self.weights) + self.bias
    y_predicted = self.Sigmoidfunction(linear_predictions)
    final_predictions = [0 if y<= 0.5 else 1 for y in y_predicted]
    #final_predictions = [1 if y > threshold else 0 for y in y_predicted]
    # print('prawdopodobieństwo:', y_predicted)
    # print('Werdykt: ', final_predictions)
    return final_predictions
  

def accuracy(y_pred, y_test):
    a = np.sum(y_pred == y_test)/len(y_test)
    print("Dokladnosc klasyfikatora wynosi: ", a)