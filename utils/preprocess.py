import pandas as pd

def load_products(csv_path="data/product.csv", max_items=1000):
    df = pd.read_csv(csv_path,on_bad_lines='skip')  # specify tab separator

    df = df.dropna(subset=['product_name', 'product_description'])
    df = df.head(max_items)
    df["text"] = df["product_name"] + " " + df["product_description"]
    return df[["product_id", "text"]]
