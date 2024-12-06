from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load the training dataset
# Replace 'path_to_training_data.csv' with the actual dataset used for model training
data = pd.read_csv('Crop_recommendation.csv')

# Select the features used for training
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Fit the MinMaxScaler and StandardScaler
ms = MinMaxScaler()
sc = StandardScaler()

ms.fit(features)  # Fit MinMaxScaler on raw features
sc.fit(ms.transform(features))  # Fit StandardScaler on MinMax-scaled features

# Save the retrained scalers
import pickle
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("Scalers retrained and saved successfully!")
