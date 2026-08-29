import python_weather as pw
from python_weather.forecast import DailyForecast as DF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from datetime import datetime, timedelta
import seaborn as sns

### Weather info tracking app

async def getDataFrame(city: str) -> pd.DataFrame:
    async with pw.Client(unit=pw.IMPERIAL) as client:
        weather = await client.get(city)
        data = []
        for daily in weather:
            for hourly in daily.hourly_forecasts:
                data.append({
                    'date': daily.date,
                    'time': hourly.time,
                    'temperature': hourly.temperature,
                    'humidity': hourly.humidity,
                    'heat_index': hourly.heat_index,
                    'wind_speed': hourly.wind_speed,
                    'precipitation': hourly.precipitation,
                    'description': hourly.description
                })
        df = pd.DataFrame(data)
        return df


async def main() -> None:

    async with pw.Client(unit=pw.IMPERIAL) as client:
        today = datetime.now()
        city = input("What city do you want weather from? ")

        weather = await client.get(city)

        for i, daily in enumerate(weather.daily_forecasts):
            daysPassed = timedelta(i)
            currDate = today + daysPassed

            print(f"The high will be {daily.highest_temperature} and the low will be {daily.lowest_temperature} on {currDate.strftime("%B %d")}")
            print(f"The wind speed on {currDate.strftime("%B %d")} will be {weather.wind_speed}mph")
            print(f"The precipitation on {currDate.strftime("%B %d")} is {weather.precipitation} \n")

        analysisQ = input("Would you like to see a plot of weather data? Y/N ")

        if analysisQ.capitalize() == "Y":
            weatherDF = await getDataFrame(city)

            plt.figure(figsize=(12, 5))
            plt.plot(weatherDF.index, weatherDF['temperature'], label='Temperature (°F)', color='orange', marker='o')
            plt.plot(weatherDF.index, weatherDF['humidity'], label='Humidity (%)', color='blue', linestyle='--')
            plt.ylabel('Value')
            plt.xlabel('Forecast Timeline Point')
            plt.title(f'Hourly Weather Forecast Trends for {city.capitalize()}')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Thank you for using the weather app!")





if __name__ == "__main__":
    asyncio.run(main())

