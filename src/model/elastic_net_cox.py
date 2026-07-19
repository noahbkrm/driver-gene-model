import pandas as pd
from data import load_cohort, DRIVER_GENES

from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from sklearn.model_selection import train_test_split

def prepare_data():

    cohort = load_cohort()

    clinical = cohort.clinical.copy()
    snv = cohort.snv.copy()
    driver_alt = cohort.driver_alt.copy()

    valid_idx = clinical.dropna(
        subset=["os_months", "os_event"]
    ).index

    clinical = clinical.loc[valid_idx]
    snv = snv.loc[valid_idx]

    snv = snv.fillna(0)

    return clinical, snv, driver_alt

def run_cox_experiment(
    X: pd.DataFrame,
    clinical: pd.DataFrame,
    name: str = "Model",
    l1_ratio: float = 0.5,
):

    print("\n======================")
    print(name)
    print("======================")

    # Create survival object
    y = Surv.from_dataframe(
        event="os_event",
        time="os_months",
        data=clinical
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    cox = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        alpha_min_ratio=0.01,
        n_alphas=50,
        max_iter=100000
    )

    cox.fit(X_train,y_train)

    '''
    coef = pd.Series(
        cox.coef_[:, -1],
        index=X_train.columns
    )

    print(
        coef.sort_values(
            key=abs,
            ascending=False
        ).head(20)
    )
    '''

    risk_scores = cox.predict(X_test)

    c_index = concordance_index_censored(
        y_test["os_event"],
        y_test["os_months"],
        risk_scores
    )

    print("C-index:", c_index[0])

    return c_index[0]


def prepare_clinical_features(clinical):

    features = clinical[
        [
            "age",
            "stage",
            "tumor_purity_pct"
        ]
    ].copy()

    # continuous variables
    features["age"] = features["age"].fillna(
        features["age"].median()
    )

    features["tumor_purity_pct"] = features["tumor_purity_pct"].fillna(
        features["tumor_purity_pct"].median()
    )

    # categorical-ish variable
    features["stage"] = features["stage"].fillna(
        features["stage"].mode()[0]
    )

    return features

def main():

    clinical, snv, driver_alt = prepare_data()


    # Experiment 1
    run_cox_experiment(
        X=snv,
        clinical=clinical,
        name="SNV only"
    )

    # Experiment 2
    clinical_features = prepare_clinical_features(clinical)

    run_cox_experiment(
        X=clinical_features,
        clinical=clinical,
        name="Clinical only"
    )


    # Experiment 3
    combined = pd.concat(
        [
            snv,
            clinical_features
        ],
        axis=1
    )

    run_cox_experiment(
        X=combined,
        clinical=clinical,
        name="SNV + Clinical"
    )

    # Experiment 4

    driver_matrix = pd.DataFrame(
        driver_alt,
        columns=DRIVER_GENES,
        index=clinical.index
    )

    run_cox_experiment(
        X=driver_matrix,
        clinical=clinical,
        name="True driver state"
    )

    # Experiment 5
    oracle = pd.concat(
        [
            driver_matrix,
            clinical_features
        ],
        axis=1
    )

    run_cox_experiment(
        X=oracle,
        clinical=clinical,
        name="Oracle drivers + clinical"
)

if __name__ == "__main__":
    main()