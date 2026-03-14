import pandas as pd


class WeatherService:
    def __init__(self, df: pd.DataFrame):
        # Save a safe copy of the dataset
        self.df = df.copy()

    def get_locations(self) -> list:
        # Return all available districts/locations
        return sorted(self.df["location"].dropna().unique().tolist())

    def get_available_days(self, location: str) -> list:
        # Filter data for the chosen location
        filtered_df = self.df[self.df["location"] == location].copy()

        # Extract only the day part from the datetime column
        available_days = filtered_df["time"].dt.date.unique().tolist()

        # Return sorted list of available days
        return sorted(available_days)

    def get_data_by_location_and_day(self, location: str, selected_day) -> pd.DataFrame:
        # Filter data for the chosen location
        filtered_df = self.df[self.df["location"] == location].copy()

        # Keep only the rows from the chosen day
        filtered_df = filtered_df[filtered_df["time"].dt.date == selected_day]

        # Sort by hour to display the day in order
        return filtered_df.sort_values("time")

# A classe WeatherService serve para selecionar e filtrar os dados meteorológicos certos consoante o distrito e o dia escolhidos na interface.