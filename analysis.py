import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def generate_sample_data(n_samples: int = 500) -> pd.DataFrame:
    """Генерация синтетического датасета заявок на кредит."""
    np.random.seed(42)
    age = np.random.randint(21, 65, size=n_samples)
    income = np.random.randint(3_000_000, 25_000_000, size=n_samples)
    loan_amount = np.random.randint(5_000_000, 80_000_000, size=n_samples)
    credit_history = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])

    debt_to_income = loan_amount / (income * 12)
    default = ((debt_to_income > 0.45) | (credit_history == 0)).astype(int)

    return pd.DataFrame({
        "age": age,
        "monthly_income": income,
        "loan_amount": loan_amount,
        "good_credit_history": credit_history,
        "debt_to_income": debt_to_income.round(2),
        "default": default,
    })


def main():
    df = generate_sample_data()

    print("=== Общая статистика по заявкам ===")
    print(df.describe().round(2))

    print("\n=== Доля дефолтов по качеству кредитной истории ===")
    print(df.groupby("good_credit_history")["default"].mean().round(2))

    X = df[["age", "monthly_income", "loan_amount", "good_credit_history", "debt_to_income"]]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n=== Метрики модели оценки риска ===")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()