import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

class Data():
    def __init__(self, path, missing_threshold=0.3, corr_threshold=0.05):
        self.raw_data = pd.read_csv(path, encoding="latin1", encoding_errors="replace")
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.selected_features = None
        self.X = None
        self.y = None
        self.scaler = None

    def obtain_data_info(self):
        # Read table shape (3047, 34)
        print("Table shape is:")
        print(self.raw_data.shape)
        print('\n')
        # Read data's variables
        print("Data variables:")
        print(self.raw_data.columns)
        print('\n')
        # Get data's variables' info like Non-Null Count and Dtype
        print(self.raw_data.info())
        print('\n')
        # Get TARGET_deathRate min and max value
        print(self.raw_data["TARGET_deathRate"].describe())
        print('\n')
        # Get Data's statistical info mean...
        print(self.raw_data.describe())
        print('\n')
        # Get missing data info
        print(self.raw_data.isnull().sum())
        print('\n')


    def feature_selection(self):
        df = self.raw_data.copy()

        # 1. drop rvariables which are not strongly relevant to Target death rate
        drop_cols = ["Geography", "binnedInc"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # 2.Deal with missing data, category data will be filled with most often appeared value,
        # numeric data will be filled with mean
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 3. use raw data to calculate missing data rate
        missing_ratio = self.raw_data.isnull().mean()

        # 4. calculation Pearson value
        corr = df.corr(numeric_only=True)["TARGET_deathRate"]

        # 5. features selections criteria
        keep_features = [
            col for col in df.columns
            if col != "TARGET_deathRate"
            and missing_ratio.get(col, 0) < self.missing_threshold
            and abs(corr.get(col, 0)) >= self.corr_threshold
        ]

        self.selected_features = keep_features
        print("\nSelected features (after filtering):\n", self.selected_features)

        self.X = df[keep_features]
        self.y = df["TARGET_deathRate"]

        return self.X, self.y, corr.sort_values(ascending=False)

    def preprocess(self):
        if self.X is None or self.y is None:
            self.feature_selection()

        # standardization
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

    X, y, corr = data.feature_selection()
    print("\nCorrelation with TARGET_deathRate:\n", corr)

    X, y = data.preprocess()
    # X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()
    #
    # print("\nTrain size:", X_train.shape, "Val size:", X_val.shape, "Test size:", X_test.shape)
