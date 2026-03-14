from services.data_loader import DataLoader

loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

print(df.head())
print()
print(df.dtypes)
