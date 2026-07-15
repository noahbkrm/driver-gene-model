import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 5],
    'B': [4, 5, 9],
})

print(df.shape)

# Sum down axis 0
result = df.mean(axis=0)
result = pd.DataFrame(result)
trans = result.T
print(result)
print(result.shape)
print(trans)
print(trans['A'])

print(np.log1p(df))

result2 = df.std(axis=0)
result2 = pd.DataFrame(result2)

print(result2)

df2 = pd.DataFrame(
    {
        "Gene_Name": ["TP53", "BRCA1", "EGFR", "MYC", "APOE"],
        "Sample_1": [2.5, np.nan, 1.8, 4.2, np.nan],
        "Sample_2": [1.9, 5.1, np.nan, np.nan, 3.3],
        "Sample_3": [np.nan, 4.8, 1.2, 3.9, 0.9],
    }
)

df2 = df2.set_index("Gene_Name")
df2 = df2.T
print(df2)

df2 = df2.fillna(df2.mean(axis = 0, numeric_only = True))

print(df2)