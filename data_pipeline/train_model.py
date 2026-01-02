import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

class ModelTrainer:
    def __init__(self, data_file="data/features.csv", model_file="data/posture_model.pkl"):
        self.data_file = data_file
        self.model_file = model_file

    def train(self):
        if not os.path.exists(self.data_file):
            print(f"Data file {self.data_file} not found. Run feature_extractor.py first.")
            return

        # Load Data
        df = pd.read_csv(self.data_file)
        
        # Clean Data
        df.dropna(inplace=True)
        
        if len(df) < 10:
            print("Not enough data to train. Please collect more samples.")
            return

        # Features (X) and Labels (y)
        # Drop filename and label
        X = df.drop(['label', 'filename'], axis=1)
        y = df['label']

        # Splitting Dataset: 70% Train, 10% Val (implied in test split), 20% Test
        # First split: 80% Train+Val, 20% Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Second split: Split Train+Val to get 10% (relative to total) Val
        # 10% of total is 12.5% of 80%.
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

        print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # mode = 'svm'
        # probability=False makes training much faster (seconds instead of minutes)
        # We will use decision_function distance as a proxy for confidence
        model = SVC(kernel='linear', probability=False) 
        # model = LogisticRegression()

        print("Training model... (Instant Mode)")
        model.fit(X_train, y_train)

        # Validation
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Test
        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, test_preds))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_preds))

        # Save Model (Main)
        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {self.model_file}")
        
        # Save Model (Backup Version)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(os.path.dirname(self.model_file), "versions")
        os.makedirs(version_dir, exist_ok=True)
        
        version_file = os.path.join(version_dir, f"model_{timestamp}.pkl")
        with open(version_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model backup saved to {version_file}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
