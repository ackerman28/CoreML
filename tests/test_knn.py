import numpy as np
from mlpackage.supervised_learning.knn import KNN

def test_knn_basic_classification():
    # 1. Setup a simple 2D dataset
    # Points near [0, 0] are Class 0
    # Points near [5, 5] are Class 1
    X_train = np.array([[0, 0], [1, 1], [0.5, 0.5], [5, 5], [6, 6], [5.5, 5.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # 2. Test a point clearly in the '0' zone
    test_point_0 = np.array([[0.2, 0.2]])
    assert knn.predict(test_point_0)[0] == 0
    
    # 3. Test a point clearly in the '1' zone
    test_point_1 = np.array([[5.2, 5.2]])
    assert knn.predict(test_point_1)[0] == 1

def test_knn_majority_vote():
    # 4. Test the 'K' logic with a tie-breaker scenario
    # Two points are Class 1, one point is Class 0
    X_train = np.array([[1, 1], [1.1, 1.1], [0, 0]])
    y_train = np.array([1, 1, 0])
    
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # The point [0.9, 0.9] is closer to the [1, 1] group
    test_point = np.array([[0.9, 0.9]])
    assert knn.predict(test_point)[0] == 1