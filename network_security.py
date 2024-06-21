import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Step 1: Generate the simulated data CSV (only if not exists)
csv_file = 'simulated_cicids2017.csv'

if not os.path.exists(csv_file):
    # Define the number of records you want to generate
    num_records = 1000

    # Generate random data for each column
    data = {
        'Destination Port': np.random.randint(0, 65535, num_records),
        'Flow Duration': np.random.randint(1, 100000, num_records),
        'Total Fwd Packets': np.random.randint(1, 1000, num_records),
        'Total Backward Packets': np.random.randint(1, 1000, num_records),
        'Total Length of Fwd Packets': np.random.uniform(0, 65535, num_records),
        'Total Length of Bwd Packets': np.random.uniform(0, 65535, num_records),
        'Fwd Packet Length Max': np.random.uniform(0, 1500, num_records),
        'Fwd Packet Length Min': np.random.uniform(0, 1500, num_records),
        'Fwd Packet Length Mean': np.random.uniform(0, 1500, num_records),
        'Fwd Packet Length Std': np.random.uniform(0, 500, num_records),
        'Bwd Packet Length Max': np.random.uniform(0, 1500, num_records),
        'Bwd Packet Length Min': np.random.uniform(0, 1500, num_records),
        'Bwd Packet Length Mean': np.random.uniform(0, 1500, num_records),
        'Bwd Packet Length Std': np.random.uniform(0, 500, num_records),
        'Flow Bytes/s': np.random.uniform(0, 1e6, num_records),
        'Flow Packets/s': np.random.uniform(0, 1e6, num_records),
        'Flow IAT Mean': np.random.uniform(0, 1e6, num_records),
        'Flow IAT Std': np.random.uniform(0, 1e6, num_records),
        'Flow IAT Max': np.random.uniform(0, 1e6, num_records),
        'Flow IAT Min': np.random.uniform(0, 1e6, num_records),
        'Fwd IAT Total': np.random.uniform(0, 1e6, num_records),
        'Fwd IAT Mean': np.random.uniform(0, 1e6, num_records),
        'Fwd IAT Std': np.random.uniform(0, 1e6, num_records),
        'Fwd IAT Max': np.random.uniform(0, 1e6, num_records),
        'Fwd IAT Min': np.random.uniform(0, 1e6, num_records),
        'Bwd IAT Total': np.random.uniform(0, 1e6, num_records),
        'Bwd IAT Mean': np.random.uniform(0, 1e6, num_records),
        'Bwd IAT Std': np.random.uniform(0, 1e6, num_records),
        'Bwd IAT Max': np.random.uniform(0, 1e6, num_records),
        'Bwd IAT Min': np.random.uniform(0, 1e6, num_records),
        'Fwd PSH Flags': np.random.randint(0, 2, num_records),
        'Bwd PSH Flags': np.random.randint(0, 2, num_records),
        'Fwd URG Flags': np.random.randint(0, 2, num_records),
        'Bwd URG Flags': np.random.randint(0, 2, num_records),
        'Fwd Header Length': np.random.randint(0, 1500, num_records),
        'Bwd Header Length': np.random.randint(0, 1500, num_records),
        'Fwd Packets/s': np.random.uniform(0, 1e6, num_records),
        'Bwd Packets/s': np.random.uniform(0, 1e6, num_records),
        'Min Packet Length': np.random.uniform(0, 1500, num_records),
        'Max Packet Length': np.random.uniform(0, 1500, num_records),
        'Packet Length Mean': np.random.uniform(0, 1500, num_records),
        'Packet Length Std': np.random.uniform(0, 500, num_records),
        'Packet Length Variance': np.random.uniform(0, 500, num_records),
        'FIN Flag Count': np.random.randint(0, 2, num_records),
        'SYN Flag Count': np.random.randint(0, 2, num_records),
        'RST Flag Count': np.random.randint(0, 2, num_records),
        'PSH Flag Count': np.random.randint(0, 2, num_records),
        'ACK Flag Count': np.random.randint(0, 2, num_records),
        'URG Flag Count': np.random.randint(0, 2, num_records),
        'CWE Flag Count': np.random.randint(0, 2, num_records),
        'ECE Flag Count': np.random.randint(0, 2, num_records),
        'Down/Up Ratio': np.random.uniform(0, 10, num_records),
        'Average Packet Size': np.random.uniform(0, 1500, num_records),
        'Avg Fwd Segment Size': np.random.uniform(0, 1500, num_records),
        'Avg Bwd Segment Size': np.random.uniform(0, 1500, num_records),
        'Fwd Header Length.1': np.random.randint(0, 1500, num_records),
        'Fwd Avg Bytes/Bulk': np.random.uniform(0, 1e6, num_records),
        'Fwd Avg Packets/Bulk': np.random.uniform(0, 1e6, num_records),
        'Fwd Avg Bulk Rate': np.random.uniform(0, 1e6, num_records),
        'Bwd Avg Bytes/Bulk': np.random.uniform(0, 1e6, num_records),
        'Bwd Avg Packets/Bulk': np.random.uniform(0, 1e6, num_records),
        'Bwd Avg Bulk Rate': np.random.uniform(0, 1e6, num_records),
        'Subflow Fwd Packets': np.random.randint(0, 1000, num_records),
        'Subflow Fwd Bytes': np.random.uniform(0, 1e6, num_records),
        'Subflow Bwd Packets': np.random.randint(0, 1000, num_records),
        'Subflow Bwd Bytes': np.random.uniform(0, 1e6, num_records),
        'Init_Win_bytes_forward': np.random.uniform(0, 65535, num_records),
        'Init_Win_bytes_backward': np.random.uniform(0, 65535, num_records),
        'act_data_pkt_fwd': np.random.randint(0, 1000, num_records),
        'min_seg_size_forward': np.random.randint(0, 1500, num_records),
        'Active Mean': np.random.uniform(0, 1e6, num_records),
        'Active Std': np.random.uniform(0, 1e6, num_records),
        'Active Max': np.random.uniform(0, 1e6, num_records),
        'Active Min': np.random.uniform(0, 1e6, num_records),
        'Idle Mean': np.random.uniform(0, 1e6, num_records),
        'Idle Std': np.random.uniform(0, 1e6, num_records),
        'Idle Max': np.random.uniform(0, 1e6, num_records),
        'Idle Min': np.random.uniform(0, 1e6, num_records),
        'Label': np.random.choice(['BENIGN', 'MALICIOUS'], num_records)
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv(csv_file, index=False)
    print("CSV file generated successfully.")
else:
    print("CSV file already exists.")

# Step 2: Model Training
model_file = 'random_forest_model.joblib'

if not os.path.exists(model_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Preprocessing: Split features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(clf, model_file)
    print("Model saved successfully.")
else:
    print("Model file already exists.")

# Step 3: Threat Detection and Alert System
# Load the trained model
clf = joblib.load(model_file)

# Load new network data (for testing, we'll use a subset of the existing data)
new_data = pd.read_csv(csv_file).sample(10)

# Preprocessing: Split features and labels
X_new = new_data.drop(columns=['Label'])
y_new = new_data['Label']

# Predict using the trained model
predictions = clf.predict(X_new)

# Initialize the language model (using GPT-2 as an example)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate recommendations
def generate_recommendation(threat_type):
    prompt = f"There is a {threat_type} detected in the network traffic. Provide recommendations to mitigate this threat."
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recommendation

# Iterate through predictions and generate alerts
for index, (prediction, true_label) in enumerate(zip(predictions, y_new)):
    if prediction == "MALICIOUS":
        threat_type = "MALICIOUS"
        recommendation = generate_recommendation(threat_type)
        print(f"Alert {index + 1}: Potential threat detected!")
        print(f"Prediction: {prediction}, Actual: {true_label}")
        print(f"Recommendation: {recommendation}")
        print("--------------------------------------------------")
    else:
        print(f"Alert {index + 1}: No threat detected.")
        print(f"Prediction: {prediction}, Actual: {true_label}")
        print("--------------------------------------------------")
