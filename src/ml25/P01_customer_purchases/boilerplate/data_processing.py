import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import ColumnTransformer
import joblib

import os
from pathlib import Path
from datetime import datetime
from ml25.P01_customer_purchases.boilerplate.negative_generation import (
    gen_all_negatives,
    gen_random_negatives,
)

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent.parent.parent / "datasets/customer_purchases/"

def clean_text_column(x):
    return x.fillna("").astype(str)

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def save_df(df, filename: str):
    # Guardar
    save_path = os.path.join(DATA_DIR, filename)
    save_path = os.path.abspath(save_path)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def extract_customer_features(df):
    """
    Extract ONLY customer demographic features (no aggregates from purchases)
    to avoid data leakage
    """
    train_df = df.copy()
    
    today = datetime.strptime("2025-21-09", "%Y-%d-%m")
    
    # Convert date columns
    train_df["customer_date_of_birth"] = pd.to_datetime(train_df["customer_date_of_birth"])
    train_df["customer_signup_date"] = pd.to_datetime(train_df["customer_signup_date"])
    
    # Calculate age and tenure
    train_df["customer_age_years"] = ((today - train_df["customer_date_of_birth"]) / pd.Timedelta(days=365)).astype(int)
    train_df["customer_tenure_years"] = ((today - train_df["customer_signup_date"]) / pd.Timedelta(days=365)).astype(int)
    
    # Get unique customer features (demographics only, no purchase aggregates)
    customer_feat = train_df[["customer_id", "customer_age_years", "customer_tenure_years"]].drop_duplicates(subset=["customer_id"])
    
    save_df(customer_feat, "customer_features.csv")
    return customer_feat

def build_processor(df, numerical_features, categorical_features, free_text_features, training=True):

    savepath = Path(DATA_DIR) / "preprocessor.pkl"
    if training:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        free_text_transformers = []
        for col in free_text_features:
            free_text_transformers.append(
                (
                    col,
                    CountVectorizer(),
                    col,
                )
            )
        # Drop label before fitting
        df_for_transform = df.drop(columns=["label"], errors="ignore")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
                *free_text_transformers,
            ],
            remainder="passthrough",  # Mantener las demas sin tocar
        )
        processed_array = preprocessor.fit_transform(df_for_transform)
        joblib.dump(preprocessor, savepath)

        # Numeric
        num_cols = numerical_features

        # Categorical
        cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )

        # Free-text
        bow_cols = []
        for col in free_text_features:
            vectorizer = preprocessor.named_transformers_[col]
            bow_cols.extend([f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()])

        # Passthrough - use df_for_transform columns
        other_cols = [
            c
            for c in df_for_transform.columns
            if c not in numerical_features + categorical_features + free_text_features
        ]

        final_cols = list(num_cols) + list(cat_cols) + bow_cols + other_cols

        processed_df = pd.DataFrame(processed_array, columns=final_cols)
        return processed_df
    else:
        # Load preprocessor for test data
        preprocessor = joblib.load(savepath)
        
        # Drop label if it exists (shouldn't be in test, but just in case)
        df_for_transform = df.drop(columns=["label"], errors="ignore")
        
        processed_array = preprocessor.transform(df_for_transform)
        
        # Reconstruct column names (same as training)
        num_cols = numerical_features
        cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        bow_cols = []
        for col in free_text_features:
            vectorizer = preprocessor.named_transformers_[col]
            bow_cols.extend([f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()])
        other_cols = [
            c for c in df_for_transform.columns
            if c not in numerical_features + categorical_features + free_text_features
        ]
        final_cols = list(num_cols) + list(cat_cols) + bow_cols + other_cols
        
        processed_df = pd.DataFrame(processed_array, columns=final_cols)
        return processed_df


def preprocess(raw_df, training=False):
    dropcols = [
        "purchase_id",
        "customer_id",
        "item_id",
        "customer_date_of_birth",
        "customer_signup_date",
        "item_release_date",
        "item_img_filename",
        "purchase_timestamp",
        "purchase_device",
        "purchase_item_rating",  # 
        "item_avg_rating",       
        "item_num_ratings",     
        "customer_item_views",   
        "label",
    ]

    # Numerical features
    numerical_features = ["item_price", "customer_age_years", "customer_tenure_years"]

    # Categorical features
    categorical_features = ["customer_gender", "item_category"]

    # Text features
    free_text_features = ["item_title"]

    # ColumnTransformer
    processed_df = build_processor(
        raw_df,
        numerical_features,
        categorical_features,
        free_text_features,
        training=training,
    )

    # Drop unnecessary columns
    processed_df = processed_df.drop(columns=dropcols, errors="ignore")
    return processed_df


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)

    # -------------- Agregar negativos ------------------ #
    # Generar negativos
    train_df_neg = gen_random_negatives(train_df, n_per_positive=1)
    train_df_neg = train_df_neg.drop_duplicates(subset=["customer_id", "item_id"])

    # Agregar Features del cliente
    train_df_cust = pd.merge(train_df, customer_feat, on="customer_id", how="left")

    # Save item_id and customer_id before preprocessing
    id_cols = train_df_cust[["item_id", "customer_id"]]
    processed_pos = preprocess(train_df_cust, training=True)
    # Add back id columns
    processed_pos = pd.concat([processed_pos, id_cols.reset_index(drop=True)], axis=1)
    processed_pos["label"] = 1

    # Obtener todas las columnas
    all_columns = processed_pos.columns

    # Separar los features exclusivos de los items (including item_id, but only once)
    item_feat = [col for col in all_columns if "item" in col]
    if "item_id" not in item_feat:
        item_feat.append("item_id")
    unique_items = processed_pos[item_feat].drop_duplicates(subset=["item_id"])

    # Separar los features exclusivos de los clientes (including customer_id, but only once)
    customer_feat = [col for col in all_columns if "customer" in col]
    if "customer_id" not in customer_feat:
        customer_feat.append("customer_id")
    unique_customers = processed_pos[customer_feat].drop_duplicates(subset=["customer_id"])

    # Agregar los features de los items a los negativos
    processed_neg = pd.merge(
        train_df_neg,
        unique_items,
        on=["item_id"],
        how="left",
    )

    # Agregar los features de los usuarios a los negativos
    processed_neg = pd.merge(
        processed_neg,
        unique_customers,
        on=["customer_id"],
        how="left",
    )

    # Agregar etiqueta a los negativos
    processed_neg["label"] = 0

    # Combinar negativos con positivos para tener el dataset completo
    processed_full = (
        pd.concat([processed_pos, processed_neg], axis=0)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    # Transformar a tipo numero
    shuffled = df_to_numeric(processed_full)
    y = shuffled["label"]

    # Eliminar columnas que no sirven
    X = shuffled.drop(columns=["label", "customer_id", "item_id"], errors="ignore")
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")
    customer_feat = read_csv("customer_features")

    # agregar features derivados del cliente al dataset
    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")

    # Procesamiento de datos
    processed = preprocess(merged, training=False)

    # Si se requiere
    dropcols = []
    processed = processed.drop(columns=dropcols)

    return df_to_numeric(processed)


if __name__ == "__main__":
    X, y = read_train_data()
    print(X.info())
    test_df = read_csv("customer_purchases_test")

    X_test = read_test_data()
    print(test_df.columns)

    test_processed = read_test_data()
