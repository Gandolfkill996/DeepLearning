import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class Data:
    def __init__(self, path, missing_threshold=0.3, corr_threshold=0.4, p_threshold=0.05):
        self.raw_data = pd.read_csv(path, encoding="latin1", encoding_errors="replace")
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.p_threshold = p_threshold
        self.selected_features = None
        self.X = None
        self.y = None
        self.scaler = None

    def obtain_data_info(self):
        print("Table shape:", self.raw_data.shape)
        print("\nData variables:", self.raw_data.columns.tolist())
        print("\nTarget statistics:\n", self.raw_data["TARGET_deathRate"].describe())
        print("\nMissing values:\n", self.raw_data.isnull().sum())

    def correlation_with_pvalues(self):
        """Compute Pearson r & p-value for each numeric column vs target"""
        results = {}
        df = self.raw_data.copy()
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            if col != "TARGET_deathRate":
                # 填充缺失再算相关
                r, p = pearsonr(df[col].fillna(df[col].mean()), df["TARGET_deathRate"])
                results[col] = (r, p)
        return pd.DataFrame(results, index=["r", "p"]).T.sort_values("r", ascending=False)

    def feature_selection(self):
        df = self.raw_data.copy()

        # Step 1: Drop not direct relevant features
        drop_cols = ["Geography", "binnedInc"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Step 2: Calculate missing rate
        missing_ratio = df.isnull().mean()
        drop_high_missing = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        df = df.drop(columns=drop_high_missing, errors="ignore")
        print("Dropped features due to high missing ratio:", drop_high_missing)

        # Step 3: fill miising value
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Step 4: calculate corr and related p value
        corr_pvals = self.correlation_with_pvalues()

        # Step 5: Filter feature with conditions
        keep_features = []
        report_rows = []
        for col, row in corr_pvals.iterrows():
            miss = missing_ratio.get(col, 0)
            r, p = row["r"], row["p"]
            keep = (miss < self.missing_threshold) and (abs(r) >= self.corr_threshold) and (p < self.p_threshold)
            if keep:
                keep_features.append(col)
            report_rows.append([col, r, p, miss, "KEEP" if keep else "DROP"])

        self.selected_features = keep_features
        self.X = df[keep_features]
        self.y = df["TARGET_deathRate"]

        # Save report
        report_df = pd.DataFrame(report_rows, columns=["Feature", "Pearson_r", "p_value", "Missing_ratio", "Decision"])
        report_df.to_csv("feature_selection_results.csv", index=False)
        print("\nFeature selection results saved to feature_selection_results.csv")

        print("\nSelected features (after filtering):\n", self.selected_features)
        return self.X, self.y, corr_pvals

    def preprocess(self):
        if self.X is None or self.y is None:
            self.feature_selection()

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        return self.X, self.y

    def split_data(self, test_size=0.2, val_size=0.2, random_state=100):
        if self.X is None or self.y is None:
            self.preprocess()

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    path = "cancer_reg-1.csv"
    data = Data(path)
    data.obtain_data_info()

    X, y, corr_pvals = data.feature_selection()
    print("\nCorrelation & P-values with TARGET_deathRate:\n", corr_pvals.head(20))

    X, y = data.preprocess()
