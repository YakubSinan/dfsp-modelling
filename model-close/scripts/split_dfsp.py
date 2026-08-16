import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/annotations.csv")

dfsp = df[df["label"] == "DFSP"].copy()

train, temp = train_test_split(
    dfsp,
    test_size=0.30,
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

train.to_csv("data/dfsp_train.csv", index=False)
val.to_csv("data/dfsp_val.csv", index=False)
test.to_csv("data/dfsp_test.csv", index=False)

print("DFSP total:", len(dfsp))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))