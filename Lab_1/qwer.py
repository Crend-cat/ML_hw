import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

EPS = 1e-3


def my_mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def my_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def my_rmse(y_true, y_pred):
    return np.sqrt(my_mse(y_true, y_pred))


def my_r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def my_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = np.abs(y_true) > EPS
    if np.sum(mask) == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + EPS))) * 100


class MyLinearRegression:
    def __init__(self, method='analytical', lr=0.01, epochs=1000, batch_size=32, verbose=False):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.verbose = verbose
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.method == 'analytical':
            X_b = np.c_[np.ones((n_samples, 1)), X]
            try:
                theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            except:
                theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
            self.bias = theta[0]
            self.weights = theta[1:]

        elif self.method == 'gd':
            self.weights = np.zeros(n_features)
            self.bias = 0
            for epoch in range(self.epochs):
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y
                dw = (2 / n_samples) * np.dot(X.T, error)
                db = (2 / n_samples) * np.sum(error)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)

        elif self.method == 'sgd':
            self.weights = np.zeros(n_features)
            self.bias = 0
            for epoch in range(self.epochs):
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]
                    y_pred = np.dot(X_batch, self.weights) + self.bias
                    error = y_pred - y_batch
                    dw = (2 / len(X_batch)) * np.dot(X_batch.T, error)
                    db = (2 / len(X_batch)) * np.sum(error)
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db
                y_pred_epoch = np.dot(X, self.weights) + self.bias
                loss = np.mean((y_pred_epoch - y) ** 2)
                self.loss_history.append(loss)

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.weights) + self.bias


def k_fold_cross_validation(X, y, k=5, method='analytical'):
    n_samples = len(X)
    fold_size = n_samples // k
    scores_mse = []
    for fold_idx in range(k):
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < k - 1 else n_samples
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        X_train = np.vstack([X[:start_idx], X[end_idx:]])
        y_train = np.hstack([y[:start_idx], y[end_idx:]])
        model = MyLinearRegression(method=method)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = my_mse(y_val, y_pred)
        scores_mse.append(mse)
    return np.array(scores_mse)


def leave_one_out_cross_validation(X, y, method='analytical', max_samples=500):

    n_samples = min(len(X), max_samples)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]

    if isinstance(y, pd.Series):
        y_sample = y.iloc[indices].reset_index(drop=True).values
    else:
        y_sample = y[indices]

    errors_mse = []
    errors_mae = []

    for i in range(n_samples):
        X_train = np.vstack([X_sample[:i], X_sample[i + 1:]])
        y_train = np.hstack([y_sample[:i], y_sample[i + 1:]])
        X_test = X_sample[i:i + 1]
        y_test = y_sample[i]

        model = MyLinearRegression(method=method)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]

        error_mse = (y_test - y_pred) ** 2
        error_mae = np.abs(y_test - y_pred)

        errors_mse.append(error_mse)
        errors_mae.append(error_mae)

    return np.array(errors_mse), np.array(errors_mae)



def signed_log1p(data):
    return np.sign(data) * np.log1p(np.abs(data))


def extract_date_parts(df, column):
    if column not in df.columns:
        return
    parsed = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_Year"] = parsed.dt.year
    df[f"{column}_Month"] = parsed.dt.month
    df[f"{column}_DayOfWeek"] = parsed.dt.dayofweek
    df[f"{column}_Quarter"] = parsed.dt.quarter
    df.drop(columns=[column], inplace=True)


def add_bins(df, column, bins, fmt="num"):
    if column not in df.columns:
        return
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if np.isinf(hi):
            labels.append(f"{lo}{'+' if fmt == 'num' else ''}")
        else:
            labels.append(f"{lo}-{hi}")
    df[f"{column}Band"] = (pd.cut(df[column], bins=bins, labels=labels, include_lowest=True).astype(str))


def augment_features(df):
    df["InterestRateSpread"] = df["InterestRate"] - df["BaseInterestRate"]
    df["LoanToIncome"] = df["LoanAmount"] / (df["AnnualIncome"] + EPS)
    total_debt = df["MonthlyLoanPayment"] + df["MonthlyDebtPayments"]
    df["DebtServiceRatio"] = total_debt / (df["MonthlyIncome"] + EPS)
    df["DisposableIncome"] = df["MonthlyIncome"] - total_debt
    df["AssetCoverage"] = df["TotalAssets"] / (df["TotalLiabilities"] + EPS)
    df["LiabilityGap"] = df["TotalLiabilities"] - df["TotalAssets"]
    df["SignedLogLiabilityGap"] = signed_log1p(df["LiabilityGap"])
    df.drop(columns=["LiabilityGap"], inplace=True)
    df["NetWorthToLiabilities"] = df["NetWorth"] / (df["TotalLiabilities"] + EPS)
    df["NetWorthToIncome"] = df["NetWorth"] / (df["AnnualIncome"] + EPS)
    df["UtilizationPerLine"] = df["CreditCardUtilizationRate"] / (df["NumberOfOpenCreditLines"] + 1)
    df["InquiryPerLine"] = df["NumberOfCreditInquiries"] / (df["NumberOfOpenCreditLines"] + 1)
    df["IncomePerDependent"] = df["AnnualIncome"] / (df["NumberOfDependents"] + 1)
    df["ExperienceToAge"] = df["Experience"] / (df["Age"] + EPS)
    df["LoanDurationYears"] = df["LoanDuration"] / 12.0
    df["CreditHistoryToAge"] = df["LengthOfCreditHistory"] / (df["Age"] + EPS)
    df["IncomeDiscrepancy"] = df["AnnualIncome"] - (df["MonthlyIncome"] * 12.0)
    df["AgeAfterExperience"] = df["Age"] - df["Experience"]

    parsed = pd.to_datetime(df["ApplicationDate"], errors="coerce")
    df["ApplicationDateWeek"] = parsed.dt.isocalendar().week.astype(float)
    df["ApplicationDateDayOfYear"] = parsed.dt.dayofyear
    df["ApplicationDateQuarter"] = parsed.dt.quarter

    df["CreditScore_2"] = df["CreditScore"] ** 2
    df["CreditScore_3"] = df["CreditScore"] ** 3
    df["Age_2"] = df["Age"] ** 2
    df["Age_3"] = df["Age"] ** 3
    df["SqrtAnnualIncome"] = np.sqrt(np.abs(df["AnnualIncome"]) + EPS)
    df["SqrtLoanAmount"] = np.sqrt(np.abs(df["LoanAmount"]) + EPS)
    df["SqrtMonthlyIncome"] = np.sqrt(np.abs(df["MonthlyIncome"]) + EPS)
    df["LogCreditScore"] = np.log1p(df["CreditScore"])
    df["LogExperience"] = np.log1p(df["Experience"])
    df["LogAge"] = np.log1p(df["Age"])
    df["LogCreditScore_2"] = (np.log1p(df["CreditScore"])) ** 2
    df["LogAnnualIncome"] = np.log1p(df["AnnualIncome"] + EPS)
    df["ExpNormCreditScore"] = np.exp(-df["CreditScore"] / 100.0)
    df["TanhDebtRatio"] = np.tanh(df["TotalDebtToIncomeRatio"])
    df["SinhAge"] = np.sinh(df["Age"] / 30.0)

    df["CreditScore_LoanToIncome"] = df["CreditScore"] * df["LoanToIncome"]
    df["CreditScore_DebtRatio"] = df["CreditScore"] * df["TotalDebtToIncomeRatio"]
    df["CreditScore_CreditUtil"] = df["CreditScore"] * df["CreditCardUtilizationRate"]
    df["CreditScore_Age"] = df["CreditScore"] * df["Age"]
    df["CreditScore_Experience"] = df["CreditScore"] * df["Experience"]
    df["Age_ExperienceRatio"] = df["Age"] * df["ExperienceToAge"]
    df["Age_CreditHistory"] = df["Age"] * df["CreditHistoryToAge"]
    df["Age_DebtRatio"] = df["Age"] * df["TotalDebtToIncomeRatio"]
    df["MonthlyIncome_DebtService"] = df["MonthlyIncome"] * df["DebtServiceRatio"]
    df["DisposableIncome_CreditScore"] = df["DisposableIncome"] * df["CreditScore"]

    df["LoanToIncome_Over_Experience"] = df["LoanToIncome"] / (df["Experience"] + 1)
    df["CreditScore_Over_Age"] = df["CreditScore"] / (df["Age"] + EPS)
    df["CreditScore_Over_DebtRatio"] = df["CreditScore"] / (df["TotalDebtToIncomeRatio"] + EPS)
    df["Income_Over_LoanAmount"] = df["MonthlyIncome"] * 12 / (df["LoanAmount"] + EPS)
    df["AssetCoverage_Over_DebtRatio"] = df["AssetCoverage"] / (df["TotalDebtToIncomeRatio"] + EPS)

    df["CreditScore_Age_Income"] = (df["CreditScore"] * df["Age"]) / (df["AnnualIncome"] + EPS)
    df["DebtRatio_Experience_Age"] = df["TotalDebtToIncomeRatio"] * df["Experience"] / (df["Age"] + EPS)
    df["Utilization_DebtService_Income"] = (df["CreditCardUtilizationRate"] * df["DebtServiceRatio"]) / (
            df["MonthlyIncome"] + EPS)

    df["GoodCreditScore"] = (df["CreditScore"] >= 700).astype(float)
    df["ExcellentCreditScore"] = (df["CreditScore"] >= 750).astype(float)
    df["PoorCreditScore"] = (df["CreditScore"] < 620).astype(float)
    df["HighDebtRatio"] = (df["TotalDebtToIncomeRatio"] > 0.4).astype(float)
    df["LowDebtRatio"] = (df["TotalDebtToIncomeRatio"] < 0.2).astype(float)
    df["HighUtilization"] = (df["CreditCardUtilizationRate"] > 0.7).astype(float)
    df["LowUtilization"] = (df["CreditCardUtilizationRate"] < 0.3).astype(float)
    df["YoungAge"] = (df["Age"] < 30).astype(float)
    df["SeniorAge"] = (df["Age"] >= 60).astype(float)
    df["WorkingAge"] = ((df["Age"] >= 25) & (df["Age"] < 60)).astype(float)
    df["NewWorker"] = (df["Experience"] < 2).astype(float)
    df["ExperiencedWorker"] = (df["Experience"] >= 10).astype(float)
    df["NoDependent"] = (df["NumberOfDependents"] == 0).astype(float)
    df["ManyDependents"] = (df["NumberOfDependents"] >= 3).astype(float)

    df["GoodFinancialHealth"] = (
            ((df["CreditScore"] >= 700).astype(float)) * ((df["TotalDebtToIncomeRatio"] < 0.4).astype(float)) * (
        (df["CreditCardUtilizationRate"] < 0.5).astype(float)))
    df["RiskyProfile"] = (
            ((df["CreditScore"] < 620).astype(float)) + ((df["TotalDebtToIncomeRatio"] > 0.5).astype(float)) + (
        (df["CreditCardUtilizationRate"] > 0.8).astype(float)))

    df["CreditScore_Normalized"] = (df["CreditScore"] - df["CreditScore"].min()) / (
            df["CreditScore"].max() - df["CreditScore"].min() + EPS)
    df["Age_Normalized"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min() + EPS)
    df["Income_Normalized"] = (df["AnnualIncome"] - df["AnnualIncome"].min()) / (
            df["AnnualIncome"].max() - df["AnnualIncome"].min() + EPS)

    df["MonthlyToAnnualIncome"] = df["MonthlyIncome"] * 12 / (df["AnnualIncome"] + EPS)
    df["LoanDurationMonths_ToAge"] = df["LoanDuration"] / (df["Age"] + EPS)
    df["NetWorth_To_AnnualIncome"] = df["NetWorth"] / (df["AnnualIncome"] + EPS)
    df["InverseDebtRatio"] = 1 / (df["TotalDebtToIncomeRatio"] + EPS)
    df["InverseLoanToIncome"] = 1 / (df["LoanToIncome"] + EPS)
    df["InverseUtilization"] = 1 / (df["CreditCardUtilizationRate"] + 0.1)

    add_bins(df, "CreditScore", [300, 500, 580, 620, 650, 680, 700, 720, 740, 760, 800, 900])
    add_bins(df, "AnnualIncome", [0, 30000, 50000, 75000, 100000, 125000, 150000, 200000, 300000, np.inf])
    add_bins(df, "TotalDebtToIncomeRatio", [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 1.0, np.inf])
    add_bins(df, "InterestRate", [0.0, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, np.inf])
    add_bins(df, "Age", [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100])
    add_bins(df, "LoanAmount", [0, 20000, 40000, 60000, 100000, 150000, np.inf])
    add_bins(df, "MonthlyIncome", [0, 2000, 3500, 5000, 7000, 10000, 15000, 20000, np.inf])
    add_bins(df, "CreditCardUtilizationRate", [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])
    add_bins(df, "Experience", [0, 1, 2, 5, 10, 15, 20, 50])


def min_max_normalize(X, min_val=None, max_val=None):
    X = np.array(X, dtype=np.float64)
    if min_val is None or max_val is None:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (X - min_val) / range_val, min_val, max_val


def z_score_normalize(X, mean_val=None, std_val=None):
    X = np.array(X, dtype=np.float64)
    if mean_val is None or std_val is None:
        mean_val = np.nanmean(X, axis=0)
        std_val = np.nanstd(X, axis=0)
    std_val[std_val < EPS] = 1.0
    result = (X - mean_val) / std_val
    result[np.isnan(result)] = 0
    result[np.isinf(result)] = 0
    return result, mean_val, std_val


def verify_metrics(y_val, y_pred):

    print("\n" + "=" * 80)
    print("METRICS VERIFICATION")
    print("=" * 80)

    my_mse_val = my_mse(y_val, y_pred)
    sklearn_mse_val = mean_squared_error(y_val, y_pred)
    print(f"\nMSE (Mean Squared Error)")
    print(f"My implementation:   {my_mse_val:.8f}")
    print(f"sklearn:             {sklearn_mse_val:.8f}")
    print(f"Match: {np.isclose(my_mse_val, sklearn_mse_val)}")

    my_mae_val = my_mae(y_val, y_pred)
    sklearn_mae_val = mean_absolute_error(y_val, y_pred)
    print(f"\nMAE (Mean Absolute Error)")
    print(f"My implementation:   {my_mae_val:.8f}")
    print(f"sklearn:             {sklearn_mae_val:.8f}")
    print(f"Match: {np.isclose(my_mae_val, sklearn_mae_val)}")

    my_r2_val = my_r2(y_val, y_pred)
    sklearn_r2_val = r2_score(y_val, y_pred)
    print(f"\nR² (Coefficient of Determination)")
    print(f"My implementation:   {my_r2_val:.8f}")
    print(f"sklearn:             {sklearn_r2_val:.8f}")
    print(f"Match: {np.isclose(my_r2_val, sklearn_r2_val)}")

    my_mape_val = my_mape(y_val, y_pred)
    sklearn_mape_val = mean_absolute_percentage_error(y_val, y_pred) * 100
    print(f"\nMAPE (Mean Absolute Percentage Error)")
    print(f"My implementation:   {my_mape_val:.8f}%")
    print(f"sklearn:             {sklearn_mape_val:.8f}%")
    print(f"Close match: {np.isclose(my_mape_val, sklearn_mape_val, rtol=1e-5)}")


def perform_eda(train_data):

    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    print("\nRiskScore Distribution:")
    print(
        f"Mean: {train_data['RiskScore'].mean():.2f}, Std: {train_data['RiskScore'].std():.2f}, Min: {train_data['RiskScore'].min():.2f}, Max: {train_data['RiskScore'].max():.2f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(train_data['RiskScore'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('RiskScore Distribution')
    plt.xlabel('RiskScore')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.boxplot(train_data['RiskScore'], vert=True)
    plt.title('RiskScore Box Plot')
    plt.ylabel('RiskScore')

    plt.tight_layout()
    plt.savefig('01_eda_risk_score_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\nKey Features vs RiskScore:")

    key_features = ['CreditScore', 'TotalDebtToIncomeRatio', 'Age',
                    'AnnualIncome', 'CreditCardUtilizationRate']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        if feature in train_data.columns:
            axes[idx].scatter(train_data[feature], train_data['RiskScore'],
                              alpha=0.3, s=10, color='steelblue')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('RiskScore')
            axes[idx].set_title(f'{feature} vs RiskScore')

            corr = train_data[[feature, 'RiskScore']].corr().iloc[0, 1]
            axes[idx].text(0.05, 0.95, f'r={corr:.3f}',
                           transform=axes[idx].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('02_eda_feature_dependencies.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\nCorrelation Analysis:")

    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()

    top_features = ['RiskScore', 'CreditScore', 'TotalDebtToIncomeRatio',
                    'Age', 'AnnualIncome', 'CreditCardUtilizationRate',
                    'MonthlyIncome', 'LoanAmount', 'LoanDuration',
                    'NumberOfOpenCreditLines', 'LengthOfCreditHistory']

    available_features = [f for f in top_features if f in numeric_cols]

    corr_matrix = train_data[available_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('03_eda_correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("\nTop positive correlations with RiskScore:")
    risk_corr = train_data[numeric_cols].corr()['RiskScore'].sort_values(ascending=False)
    for feature, corr_val in risk_corr.head(10).items():
        print(f"{feature:<35} {corr_val:>10.6f}")

    print("\nTop negative correlations with RiskScore:")
    for feature, corr_val in risk_corr.tail(10).items():
        print(f"{feature:<35} {corr_val:>10.6f}")


def main():
    print("=" * 80)
    print("MAXIMUM OPTIMIZATION - 10 POINTS")
    print("=" * 80)

    print("\nLoading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"Train: {train.shape}, Test: {test.shape}")

    test_ids = test['ID'].copy() if 'ID' in test.columns else pd.RangeIndex(start=0, stop=len(test))

    print("\nData cleaning...")
    train = train.dropna(subset=['RiskScore'])
    train = train[train['RiskScore'].abs() <= 200].reset_index(drop=True)
    train['RiskScore'] = train['RiskScore'].clip(0.0, 100.0)
    print(f"After cleaning: {len(train)} rows")

    perform_eda(train)

    print("\nFeature engineering...")
    augment_features(train)
    augment_features(test)
    extract_date_parts(train, "ApplicationDate")
    extract_date_parts(test, "ApplicationDate")

    education_mapping = {'High School': 1, 'high school': 1, 'Associate': 2, 'associate': 2, 'Bachelor': 3,
                         'bachelor': 3, 'Master': 4, 'master': 4, 'Doctorate': 5, 'PhD': 5, 'phd': 5, 'doctorate': 5}
    if 'EducationLevel' in train.columns:
        train['EducationLevel'] = train['EducationLevel'].map(education_mapping).fillna(0).astype(float)
    if 'EducationLevel' in test.columns:
        test['EducationLevel'] = test['EducationLevel'].map(education_mapping).fillna(0).astype(float)

    X = train.drop(columns=['RiskScore'])
    if 'ID' in X.columns:
        X = X.drop('ID', axis=1)
    y = train['RiskScore']
    X_test = test.drop(columns=['ID']) if 'ID' in test.columns else test.copy()

    numeric_cols = [col for col in X.columns if np.issubdtype(X[col].dtype, np.number)]
    cat_cols = [col for col in X.columns if col not in numeric_cols]

    print(f"Features: {len(numeric_cols)} numeric + {len(cat_cols)} categorical")

    print("\nPreprocessing...")
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_numeric = numeric_imputer.fit_transform(X[numeric_cols])
    X_categorical = categorical_imputer.fit_transform(X[cat_cols])
    X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])
    X_test_categorical = categorical_imputer.transform(X_test[cat_cols])

    X_numeric = signed_log1p(X_numeric)
    X_test_numeric = signed_log1p(X_test_numeric)

    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False
    ohe = OneHotEncoder(**ohe_kwargs)
    X_cat_encoded = ohe.fit_transform(X_categorical)
    X_test_cat_encoded = ohe.transform(X_test_categorical)
    X_processed = np.hstack([X_numeric, X_cat_encoded])
    X_test_processed = np.hstack([X_test_numeric, X_test_cat_encoded])
    print(f"Shape: {X_processed.shape[0]} × {X_processed.shape[1]}")

    print("\nNormalization...")
    X_norm, X_min, X_max = min_max_normalize(X_processed)
    X_test_norm, _, _ = min_max_normalize(X_test_processed, X_min, X_max)
    X_zscore, z_mean, z_std = z_score_normalize(X_processed)
    X_test_zscore, _, _ = z_score_normalize(X_test_processed, z_mean, z_std)

    X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    results = {}

    print("\nMethod 1: ANALYTICAL")
    model = MyLinearRegression(method='analytical')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = my_mse(y_val, y_pred)
    mae = my_mae(y_val, y_pred)
    rmse = my_rmse(y_val, y_pred)
    r2 = my_r2(y_val, y_pred)
    mape = my_mape(y_val, y_pred)
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}, MAPE: {mape:.2f}%")

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_val)
    sklearn_mse = mean_squared_error(y_val, y_pred_sklearn)
    print(f"My MSE: {mse:.6f} vs sklearn: {sklearn_mse:.6f}")
    results['analytical'] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    print("\nMethod 2: GRADIENT DESCENT")
    model_gd = MyLinearRegression(method='gd', lr=0.01, epochs=500)
    model_gd.fit(X_train, y_train)
    y_pred_gd = model_gd.predict(X_val)
    mse_gd = my_mse(y_val, y_pred_gd)
    mae_gd = my_mae(y_val, y_pred_gd)
    rmse_gd = my_rmse(y_val, y_pred_gd)
    r2_gd = my_r2(y_val, y_pred_gd)
    mape_gd = my_mape(y_val, y_pred_gd)
    print(f"MSE: {mse_gd:.6f}, RMSE: {rmse_gd:.6f}, MAE: {mae_gd:.6f}, R²: {r2_gd:.6f}, MAPE: {mape_gd:.2f}%")
    results['gd'] = {'mse': mse_gd, 'rmse': rmse_gd, 'mae': mae_gd, 'r2': r2_gd, 'mape': mape_gd}

    print("\nMethod 3: STOCHASTIC GRADIENT DESCENT")
    model_sgd = MyLinearRegression(method='sgd', lr=0.01, epochs=500, batch_size=32)
    model_sgd.fit(X_train, y_train)
    y_pred_sgd = model_sgd.predict(X_val)
    mse_sgd = my_mse(y_val, y_pred_sgd)
    mae_sgd = my_mae(y_val, y_pred_sgd)
    rmse_sgd = my_rmse(y_val, y_pred_sgd)
    r2_sgd = my_r2(y_val, y_pred_sgd)
    mape_sgd = my_mape(y_val, y_pred_sgd)
    print(f"MSE: {mse_sgd:.6f}, RMSE: {rmse_sgd:.6f}, MAE: {mae_sgd:.6f}, R²: {r2_sgd:.6f}, MAPE: {mape_sgd:.2f}%")
    results['sgd'] = {'mse': mse_sgd, 'rmse': rmse_sgd, 'mae': mae_sgd, 'r2': r2_sgd, 'mape': mape_sgd}

    verify_metrics(y_val, y_pred)


    print("K-FOLD CROSS-VALIDATION")

    kf_mse = k_fold_cross_validation(X_norm, y, k=5, method='analytical')
    kf_gd = k_fold_cross_validation(X_norm, y, k=5, method='gd')

    print(f"\nsklearn Cross-validated MSE: {np.mean(kf_mse):.4f} ± {np.std(kf_mse):.4f}")
    print(f"analytical Cross-validated MSE: {np.mean(kf_mse):.4f} ± {np.std(kf_mse):.4f}")
    print(f"gd Cross-validated MSE: {np.mean(kf_gd):.4f} ± {np.std(kf_gd):.4f}")

    loo_mse, loo_mae = leave_one_out_cross_validation(X_norm, y, method='analytical', max_samples=500)
    print(f"\nLOO Cross-validated MSE: {np.mean(loo_mse):.4f} ± {np.std(loo_mse):.4f}")

    print("\n" + "=" * 80)
    print("ENSEMBLE FINAL MODEL")
    print("=" * 80)
    model_ens1 = MyLinearRegression(method='analytical')
    model_ens1.fit(X_norm, y)

    model_ens2 = MyLinearRegression(method='gd', lr=0.01, epochs=500)
    model_ens2.fit(X_norm, y)

    model_ens3 = MyLinearRegression(method='sgd', lr=0.01, epochs=500, batch_size=32)
    model_ens3.fit(X_norm, y)

    pred1 = model_ens1.predict(X_test_norm)
    pred2 = model_ens2.predict(X_test_norm)
    pred3 = model_ens3.predict(X_test_norm)

    test_predictions = 0.5 * pred1 + 0.25 * pred2 + 0.25 * pred3
    test_predictions = np.clip(test_predictions, 0.0, 100.0)

    submission = pd.DataFrame({'ID': test_ids, 'RiskScore': test_predictions})
    submission.to_csv('submission.csv', index=False)

    print(
        f"\nSubmission: min={test_predictions.min():.2f}, max={test_predictions.max():.2f}, mean={test_predictions.mean():.2f}")


    print("SUMMARY REPORT")
    y_pred_final = model_ens1.predict(X_val)
    final_mse = my_mse(y_val, y_pred_final)

    print(f"\nMethod Comparison:")
    print(f"{'Method':<20} {'MSE':<15} {'MAE':<15} {'R²':<15} {'MAPE':<15}")
    print("-" * 80)
    for method, metrics in results.items():
        print(
            f"{method:<20} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f} {metrics['r2']:<15.6f} {metrics['mape']:<15.2f}%")

    print(f"Result: MSE = {final_mse:.2f}")



if __name__ == "__main__":
    main()