import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def plot_decision_boundary(feature_a : pd.Series,
                           feature_b : pd.Series,
                           target : pd.Series,
                           threshold : float = 0.5
                          ):

    X = pd.concat([feature_a,feature_b],axis=1)
    model = LogisticRegression()
    model.fit(X,target)

    xx, yy = np.meshgrid(np.linspace(feature_a.min() - 1, feature_a.max() +1, 100), 
                         np.linspace(feature_b.min()-1, feature_b.max()+1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contourf(xx, yy, probs, levels=[0, threshold, 1], cmap='coolwarm', alpha=0.8)
    plt.scatter(feature_a, feature_b, c=target, edgecolors='k', marker='o', s=50)
    plt.xlabel(f'{feature_a.name}')
    plt.ylabel(f'{feature_b.name}')
    plt.title('Decision Boundary of the Logistic Regression')
    plt.show()


