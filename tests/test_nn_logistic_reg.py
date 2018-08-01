import sys
sys.path.append("../")
from kleinml.neural_network import LogisticRegression
from kleinml.decomposition import PCA
from kleinml.utils import load_data
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

train_x, train_y, test_x, test_y = load_data.load_iris()
pca = PCA(n_components=2)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
model = LogisticRegression(max_iter=1000, batch_size=32)
model.fit(train_x, train_y)
pred = model.predict(test_x)
plot_decision_regions(train_x, train_y, clf=model, legend=2)
plt.title('Logistic regression on diabetes dataset')
plt.show()
