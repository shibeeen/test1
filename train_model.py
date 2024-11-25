import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
data_path = 'online_shopping.csv'  # Update with the correct path to your dataset
data = pd.read_csv(data_path)

# Rename columns for easier access (optional)
data.columns = ['Timestamp', 'Shopping_Frequency', 'Age_Group', 'Electronics_Platform',
                'Fashion_Platform', 'Beauty_Platform', 'Grocery_Platform', 
                'Important_Factor', 'Trust_Reviews', 'Best_Return_Policy']

# Drop the 'Timestamp' column as it's not relevant for training
data = data.drop(columns=['Timestamp'])

# Drop rows with missing values (if any)
data = data.dropna()

# Define the feature columns (excluding the target column)
X = data.drop(columns=['Important_Factor'])

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Define the target variable (what we want to predict)
y = data['Important_Factor']

# Encode the target variable (converting text labels into numbers)
y = pd.factorize(y)[0]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to a file (optional)
joblib.dump(model, 'online_shopping_model.pkl')
