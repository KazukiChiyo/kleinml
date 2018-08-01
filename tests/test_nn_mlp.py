import sys
sys.path.append("../")
from kleinml.neural_network import MLPClassifier
from kleinml.utils import load_data
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

train_x, train_y, test_x, test_y = load_data.load_rbf()

model = MLPClassifier(hidden_layer_shapes=(16, 32, 16), activation="relu", max_iter=1000, batch_size=32)
model.fit(train_x, train_y)
pred = model.predict(test_x)
plot_decision_regions(train_x, train_y, clf=model, legend=2)
plt.title('')
plt.show()
