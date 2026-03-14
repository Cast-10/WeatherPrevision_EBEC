import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)

        # Remove spaces before and after column names
        df.columns = df.columns.str.strip()

        # Convert time column to datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        return df


# A classe DataLoader serve para carregar o ficheiro CSV e preparar os dados para serem usados corretamente pela interface.