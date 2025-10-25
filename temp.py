import pandas as pd


train_df = pd.read_csv('BradleyMeltingPointDataset.csv')
test_df = pd.read_csv('test.csv')

cols = ['smiles', 'mpC']
df = train_df[cols].copy()
df['Tm'] = df['mpC'] + 273.15
df.drop('mpC', axis=1, inplace=True)
df = df.rename(columns={'smiles': 'SMILES'})

# Build a set of SMILES in test_df
test_smiles = set(test_df['SMILES'].astype(str).str.strip())

# Filter df to keep only rows not in test set
before = len(df)
df = df[~df['SMILES'].astype(str).str.strip().isin(test_smiles)].copy()
print(f"Removed {before - len(df)} overlap rows from df")


df.to_csv('train_2.csv', index=False)



