import pandas as pd
import numpy as np

# Rastgele veri üretme
np.random.seed(42)  # Aynı veriyi tekrar üretmek için
square_meters = np.random.randint(50, 250, size=1000)  # 50-250 m² arasında 100 örnek
price = square_meters * np.random.randint(2000, 5000) + np.random.randint(50000, 100000, size=1000)

# DataFrame oluştur
dataset = pd.DataFrame({"SquareMeters": square_meters, "Price": price})

# CSV olarak kaydet
dataset.to_csv("dataset.csv", index=False)

print("dataset.csv dosyası oluşturuldu!")
