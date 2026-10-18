"""The three public tabular datasets used in the study, with protected attributes excluded."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from esslib import Dataset

CACHE = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "recruitment": "https://raw.githubusercontent.com/devashishpatel/HR-Analytics/master/HR_Data_No_Missing_Value.csv",
    "adult": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
    "german": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv",
}


def _read(name: str, **kw) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.csv"
    if not path.exists():
        pd.read_csv(SOURCES[name], **kw).to_csv(path, index=False)
    return pd.read_csv(path)


def recruitment() -> Dataset:
    """HR offer-acceptance data used in the dissertation (8,995 candidates)."""
    df = _read("recruitment")
    excluded = ["Gender"]
    df = df.drop(columns=["SLNO", "Candidate.Ref", *excluded])
    y = df.pop("Status").map({"Not Joined": 0, "Joined": 1})
    cat = df.select_dtypes(include="object").columns.tolist()
    return Dataset("Recruitment", df, y, cat, excluded)


def adult(n: int = 20000, seed: int = 0) -> Dataset:
    """UCI Adult census income, subsampled for compute; race and sex excluded."""
    cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"]
    df = _read("adult", header=None, names=cols, skipinitialspace=True, na_values="?")
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    excluded = ["race", "sex"]
    df = df.drop(columns=["fnlwgt", *excluded])
    y = df.pop("income").astype(str).str.contains(">50K").astype(int)
    cat = df.select_dtypes(include="object").columns.tolist()
    return Dataset("Adult", df, y, cat, excluded)


def german() -> Dataset:
    """Statlog German credit (1,000 applicants); the combined personal status and sex attribute excluded."""
    cols = ["checking", "duration", "credit_history", "purpose", "amount", "savings", "employment",
            "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property", "age",
            "other_installments", "housing", "existing_credits", "job", "dependents", "telephone",
            "foreign_worker", "target"]
    df = _read("german", header=None, names=cols)
    excluded = ["personal_status_sex"]
    df = df.drop(columns=excluded)
    y = (df.pop("target") == 1).astype(int)  # 1 = good credit
    cat = df.select_dtypes(include="object").columns.tolist()
    return Dataset("German credit", df, y, cat, excluded)


ALL = {"recruitment": recruitment, "adult": adult, "german": german}
