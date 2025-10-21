import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr


class Data:
    def __init__(self, path, missing_threshold=0.3, corr_threshold=0.2, p_threshold=0.05,
                 use_pca=False, n_components=None, use_l1=False):
        """
        Load and preprocess dataset.
        Works for both training data (with TARGET_deathRate) and
        new data (without TARGET_deathRate).
        """
        self.raw_data = pd.read_csv(path, encoding="latin1", encoding_errors="replace")
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.p_threshold = p_threshold
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_l1 = use_l1
        self.selected_features = None
        self.X = None
        self.y = None
        self.scaler = None

    def correlation_with_pvalues(self):
        """
        Compute correlation and p-values between features and target.
        Only works if TARGET_deathRate exists in dataset.
        """
        results = {}
        df = self.raw_data.copy()
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            if col != "TARGET_deathRate":
                r, p = pearsonr(df[col].fillna(df[col].mean()), df["TARGET_deathRate"])
                results[col] = (r, p)
        return pd.DataFrame(results, index=["r", "p"]).T.sort_values("r", ascending=False)

    def feature_selection(self):
        """
        Select features for training or prediction.
        - If TARGET_deathRate exists → do correlation-based filtering
        - If not → use all numeric columns for prediction
        """
        df = self.raw_data.copy()
        drop_cols = ["Geography", "binnedInc"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        missing_ratio = df.isnull().mean()

        # Case 1: Training data

        if "TARGET_deathRate" in df.columns:
            corr_pvals = self.correlation_with_pvalues()
            keep_features = [
                col for col, row in corr_pvals.iterrows()
                if missing_ratio.get(col, 0) < self.missing_threshold
                and abs(row["r"]) >= self.corr_threshold
                and row["p"] < self.p_threshold
            ]

            self.selected_features = keep_features
            self.X = df[keep_features]
            self.y = df["TARGET_deathRate"]

            # PCA
            if self.use_pca and self.n_components:
                pca = PCA(n_components=self.n_components)
                self.X = pca.fit_transform(self.X)
                self.selected_features = [f"PCA_{i}" for i in range(self.n_components)]

            # L1 feature selection
            if self.use_l1:
                lasso = LassoCV(cv=5).fit(self.X, self.y)
                mask = lasso.coef_ != 0
                self.X = self.X[:, mask]
                self.selected_features = [f for f, m in zip(self.selected_features, mask) if m]

            return self.X, self.y, corr_pvals


        # Case 2: New data (no target)

        else:
            keep_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            self.selected_features = keep_features
            self.X = df[keep_features]
            self.y = None
            return self.X, None, None

    def preprocess(self):
        """
        Scale features after selection.
        """
        if self.X is None or self.y is None:
            self.feature_selection()

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        return self.X, self.y

    def split_data(self, test_size=0.2, val_size=0.2, random_state=100):
        """
        Split into train/val/test. Only works if target exists.
        """
        if self.X is None or self.y is None:
            self.preprocess()

        if self.y is None:
            raise ValueError("No TARGET_deathRate column found — cannot split data for training.")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


