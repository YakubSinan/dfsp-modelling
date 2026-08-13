from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "annotations.csv")

# Sadece Dermatofibroma
df = df[df["label"] == "DF"].copy()

print(f"DF total: {len(df)}")

# 70% train, 15% validation, 15% test
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

train.to_csv(ROOT / "data" / "df_train.csv", index=False)
val.to_csv(ROOT / "data" / "df_val.csv", index=False)
test.to_csv(ROOT / "data" / "df_test.csv", index=False)

print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")

print("\nDF split tamamlandı.")