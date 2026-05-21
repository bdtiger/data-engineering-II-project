from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
dataset = loadtxt('github-repository-data.csv', delimiter=',')

# split into X and y
X = dataset[:,0:8]
y = dataset[:,8]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# create model
model = LogisticRegression(max_iter=1000)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")