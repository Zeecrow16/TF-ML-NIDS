import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class CwtTraining:
    """
    Train ML models (Logistic Regression, KNN, Random Forest) on cwt processed dataset.
    """

    def __init__(
        self,
        csv_path: str,
        label_col: str = "Label",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.csv_path = csv_path
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.trained_models = {}

    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)

        drop_cols = (
            [self.label_col, "Attack"] if "Attack" in df.columns else [self.label_col]
        )
        X = df.drop(columns=drop_cols)
        y = df[self.label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.trained_models["Logistic Regression"] = model

    def train_knn(self, n_neighbors: int = 5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self.X_train, self.y_train)
        self.trained_models["KNN"] = model

    def train_random_forest(self, n_estimators: int = 200):
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.trained_models["Random Forest"] = model
