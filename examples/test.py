import sys
sys.path.append("../")
from kleinml.neural_network import NeuralNetwork
from kleinml.neural_network.layers import Dense, Activation
from kleinml.decomposition import PCA
from kleinml.utils import load_data
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

train_x, train_y, test_x, test_y = load_data.load_iris()
pca = PCA(n_components=2)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
n_features = train_x.shape[1]
model = NeuralNetwork(max_iter=100, batch_size=32)
model.add(Dense(32, input_shape=(n_features, )))
model.add(Activation("relu"))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(3))
model.add(Activation("softmax"))
model.fit(train_x, train_y)
pred = model.predict(test_x)
print(test_y, pred)
plot_decision_regions(train_x, train_y, clf=model, legend=2)
plt.title('Logistic regression on diabetes dataset')
plt.show()
