"""Module for preprocessing the rings prediction dataset."""

import pandas as pd
from prefect import task
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@task(name="preprocess_data_task")
def preprocess_data_task(df):
    """Preprocess the input DataFrame for model training.

    Performs one-hot encoding, outlier removal, feature scaling, and feature
    selection on the input data to prepare it for model training.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.

    Returns:
        tuple: A tuple containing the training and testing sets for
            features (x_train, x_test) and target (y_train, y_test).
    """
    # Perform one-hot encoding
    data = pd.get_dummies(df)

    # Outlier removal
    outlier_conditions(data)

    # Define features and target
    x = data.drop("Rings", axis=1)
    y = data["Rings"]

    # Scaling
    standard_scale = StandardScaler()
    x_scaled = standard_scale.fit_transform(x)

    # Feature selection
    select_k_best = SelectKBest()
    xn = select_k_best.fit_transform(x_scaled, y)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(xn, y, test_size=0.25)

    return x_train, x_test, y_train, y_test


def outlier_conditions(data):
    """Remove outliers from the dataset based on specific conditions.

    Removes outliers based on relationships between 'Viscera weight',
    'Shell weight', 'Shucked weight', 'Diameter', 'Height' and 'Rings'.

    Args:
        data (pd.DataFrame): The input DataFrame from which outliers
            will be removed.

    Returns:
        pd.DataFrame: The cleaned DataFrame with outliers removed.
    """
    # Outlier removal based on the 'Viscera weight'
    data.drop(
        data[(data["Viscera weight"] > 0.5) & (data["Rings"] < 20)].index,
        inplace=True,
    )
    data.drop(
        data[(data["Viscera weight"] < 0.5) & (data["Rings"] > 25)].index,
        inplace=True,
    )

    # Outlier removal based on the 'Shell weight'
    data.drop(
        data[(data["Shell weight"] > 0.6) & (data["Rings"] < 25)].index,
        inplace=True,
    )
    data.drop(
        data[(data["Shell weight"] < 0.8) & (data["Rings"] > 25)].index,
        inplace=True,
    )

    # Outlier removal based on the 'Shucked weight'
    data.drop(
        data[(data["Shucked weight"] >= 1) & (data["Rings"] < 20)].index,
        inplace=True,
    )
    data.drop(
        data[(data["Shucked weight"] < 1) & (data["Rings"] > 20)].index,
        inplace=True,
    )

    # Outlier removal based on the 'Diameter'
    data.drop(
        data[(data["Diameter"] < 0.1) & (data["Rings"] < 5)].index,
        inplace=True,
    )
    data.drop(
        data[(data["Diameter"] < 0.6) & (data["Rings"] > 25)].index,
        inplace=True,
    )
    data.drop(
        data[(data["Diameter"] >= 0.6) & (data["Rings"] < 25)].index,
        inplace=True,
    )

    return data
