import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data_path = "datasetnew.csv"  # Ensure the path is correct
data = pd.read_csv(data_path)

# Prepare the data
features = data[['Skills Required', 'Location', 'Eligibility', 'Duration (Months)']]
target_title = data['Internship Title']
target_description = data['Internship Description']
target_stipend = data['Stipend (INR/Month)']

# Encode categorical features if necessary
features = pd.get_dummies(features)

# Encode the target columns
encoder_title = LabelEncoder()
encoded_title = encoder_title.fit_transform(target_title)

encoder_description = LabelEncoder()
encoded_description = encoder_description.fit_transform(target_description)

# Split the data for each target
X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(features, encoded_title, test_size=0.2, random_state=42)
X_train_desc, X_test_desc, y_train_desc, y_test_desc = train_test_split(features, encoded_description, test_size=0.2, random_state=42)
X_train_stipend, X_test_stipend, y_train_stipend, y_test_stipend = train_test_split(features, target_stipend, test_size=0.2, random_state=42)

# Train the models
model_title = RandomForestRegressor()
model_title.fit(X_train_title, y_train_title)

model_description = RandomForestRegressor()
model_description.fit(X_train_desc, y_train_desc)

model_stipend = RandomForestRegressor()
model_stipend.fit(X_train_stipend, y_train_stipend)

# Save the models and encoders using pickle
with open("model_title.pkl", "wb") as file:
    pickle.dump(model_title, file)

with open("model_description.pkl", "wb") as file:
    pickle.dump(model_description, file)

with open("model_stipend.pkl", "wb") as file:
    pickle.dump(model_stipend, file)

with open("encoder_title.pkl", "wb") as file:
    pickle.dump(encoder_title, file)

with open("encoder_description.pkl", "wb") as file:
    pickle.dump(encoder_description, file)
