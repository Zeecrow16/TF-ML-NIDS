import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class BaselineTraining:
    """
    Train baseline models (Logistic Regression, KNN, Random Forest) using raw features from UNSW-NB15 dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        csv_path: str = None,
        label_col: str = "Label",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = df
        self.csv_path = csv_path
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.trained_models = {}

    def load_and_prepare_data(self, n_rows: int = None):
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)

        if "Label" not in self.df.columns:
            if "Attack" in self.df.columns:
                self.df["Label"] = (self.df["Attack"] != "Benign").astype(int)
        else:
            raise KeyError("Dataframe must have 'Attack' column to create 'Label'")

        if n_rows is not None:
            self.df = self.df.sample(n=n_rows, random_state=self.random_state)

        drop_cols = [
            "Attack",
            "Label",
            "IPV4_SRC_ADDR",
            "IPV4_DST_ADDR",
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
        ]

        X = self.df.drop(columns=[col for col in drop_cols if col in self.df.columns])
        y = self.df[self.label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train_scaled, self.y_train)
        self.trained_models["Logistic Regression"] = model

    def train_knn(self):
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X_train_scaled, self.y_train)
        self.trained_models["KNN"] = model

    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.trained_models["Random Forest"] = model
