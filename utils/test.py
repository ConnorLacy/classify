 # Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # load the iris datasets
    dataset = datasets.load_iris()
    correlated_features = set()
    correlation_matrix = dataset.corr()
    # create a base classifier used to evaluate a subset of attributes
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 3)
    rfe = rfe.fit(dataset.data, dataset.target)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)