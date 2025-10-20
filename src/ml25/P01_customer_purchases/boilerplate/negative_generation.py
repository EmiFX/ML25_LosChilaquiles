import numpy as np
import pandas as pd
import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def gen_smart_negatives(df, n_per_positive=2):
    """
    Generate smart negatives based on intelligent heuristics:
    - Sample from categories the customer has browsed but not purchased
    - Sample items with similar price ranges to what customer bought
    - Mix of popular items (harder negatives) and random items
    """
    negatives = get_negatives(df)
    negative_lst = []
    
    # Get price statistics per customer
    customer_price_stats = df.groupby("customer_id")["item_price"].agg(['mean', 'std']).to_dict('index')
    
    # Get customer's purchased categories
    customer_categories = df.groupby("customer_id")["item_category"].apply(set).to_dict()
    
    # Get all items with their categories and prices
    item_info = df[["item_id", "item_category", "item_price"]].drop_duplicates(subset=["item_id"])
    item_dict = item_info.set_index("item_id").to_dict('index')
    
    # Get popular items (by purchase frequency)
    item_popularity = df["item_id"].value_counts().to_dict()
    
    for customer_id, item_set in negatives.items():
        if len(item_set) == 0:
            continue
            
        available_items = list(item_set)
        n_samples = min(n_per_positive, len(available_items))
        
        # Strategy: Mix of different negative types
        n_similar_price = max(1, n_samples // 2)  # Half from similar price range
        n_popular = max(1, n_samples // 3)  # Some popular items (harder negatives)
        n_random = n_samples - n_similar_price - n_popular  # Rest random
        
        selected_items = []
        
        # 1. Sample items with similar prices to customer's purchases
        if customer_id in customer_price_stats:
            mean_price = customer_price_stats[customer_id]['mean']
            std_price = customer_price_stats[customer_id].get('std', mean_price * 0.2)
            
            price_similar_items = [
                item_id for item_id in available_items
                if item_id in item_dict and 
                abs(item_dict[item_id]['item_price'] - mean_price) <= std_price * 1.5
            ]
            
            if len(price_similar_items) >= n_similar_price:
                selected_items.extend(np.random.choice(price_similar_items, size=n_similar_price, replace=False))
            else:
                selected_items.extend(price_similar_items)
        
        # 2. Sample popular items (harder negatives - items many people buy but this customer didn't)
        popular_items = [
            item_id for item_id in available_items
            if item_id in item_popularity and item_id not in selected_items
        ]
        popular_items.sort(key=lambda x: item_popularity.get(x, 0), reverse=True)
        
        n_popular_to_add = min(n_popular, len(popular_items))
        if n_popular_to_add > 0:
            selected_items.extend(popular_items[:n_popular_to_add])
        
        # 3. Fill remaining with random items
        remaining_items = [item for item in available_items if item not in selected_items]
        n_random_needed = n_samples - len(selected_items)
        
        if n_random_needed > 0 and len(remaining_items) > 0:
            n_random_to_add = min(n_random_needed, len(remaining_items))
            selected_items.extend(np.random.choice(remaining_items, size=n_random_to_add, replace=False))
        
        # Create negative samples
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in selected_items[:n_samples]
        ]
        negative_lst.extend(negatives_for_customer)
    
    return pd.DataFrame(negative_lst)

def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())

    negatives = {}
    for customer in unique_customers:
        purcharsed_items = df[df["customer_id"] == customer]["item_id"].unique()
        non_purchased = unique_items - set(purcharsed_items)
        negatives[customer] = non_purchased
    return negatives


def gen_all_negatives(df):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in item_set
        ]
        negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)


def gen_random_negatives(df, n_per_positive=2):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        rand_items = np.random.choice(list(item_set), size=n_per_positive)
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in rand_items
        ]
        negative_lst.extend(negatives_for_customer)
    neg_df = pd.DataFrame(negative_lst)
    return neg_df


def gen_final_dataset(train_df, negatives):
    customer_columns = [
        "customer_id",
        "customer_date_of_birth",
        "customer_gender",
        "customer_signup_date",
    ]

    item_columns = [
        "item_id",
        "item_title",
        "item_category",
        "item_price",
        "item_img_filename",
        "item_avg_rating",
        "item_num_ratings",
        "item_release_date",
    ]

    # Get unique customer info
    customer_info = train_df[customer_columns].drop_duplicates(subset=["customer_id"])
    
    # Get unique item info
    item_info = train_df[item_columns].drop_duplicates(subset=["item_id"])
    
    # Merge negatives with customer and item information
    negatives_with_info = negatives.merge(customer_info, on="customer_id", how="left")
    negatives_with_info = negatives_with_info.merge(item_info, on="item_id", how="left")
    
    # Add label to positive samples
    train_df_labeled = train_df.copy()
    train_df_labeled["label"] = 1
    
    # Concatenate positive and negative samples
    final_df = pd.concat([train_df_labeled, negatives_with_info], axis=0, ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return final_df


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")

    allnegatives = gen_all_negatives(train_df)
    print(allnegatives.info())
    
    randnegatives = gen_random_negatives(train_df, n_per_positive=3)
    print(randnegatives.info())
    
    smartnegatives = gen_smart_negatives(train_df, n_per_positive=3)
    print(smartnegatives.info())
