import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Data preprocessing function
def preprocess_data(df):
    # Convert skill levels to numerical values with more nuanced scaling
    skill_mapping = {
        'Not Interested': 0,
        'Poor': 2,
        'Beginner': 4,
        'Average': 6,
        'Intermediate': 8,
        'Professional': 10,
        'Excellent': 12
    }
    
    # Apply mapping to all skill columns
    feature_cols = df.columns[:-1]  # All columns except 'Role'
    df[feature_cols] = df[feature_cols].replace(skill_mapping)
    
    # Encode the target variable
    le = LabelEncoder()
    df['Role'] = le.fit_transform(df['Role'])
    
    return df, le

# Model creation and training function
def create_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Increased number of trees
        max_depth=None,  # Allow deeper trees
        min_samples_split=5,  # Increased to reduce overfitting
        min_samples_leaf=2,  # Prevent creating too specific leaf nodes
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced',  # Ensure fair treatment of all classes
        criterion='entropy'  # Use entropy for information gain
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

# Career prediction function
def predict_careers(skills_instance):
    # Load the trained model and label encoder
    with open('advisor/ml_models/career_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('advisor/ml_models/label_encoder.pkl', 'rb') as encoder_file:
        le = pickle.load(encoder_file)
    
    feature_names = [
        'Database Fundamentals', 'Computer Architecture',
        'Distributed Computing Systems', 'Cyber Security', 'Networking',
        'Software Development', 'Programming Skills', 'Project Management',
        'Computer Forensics Fundamentals', 'Technical Communication', 
        'AI ML', 'Software Engineering', 'Business Analysis', 
        'Communication skills', 'Data Science', 
        'Troubleshooting skills', 'Graphics Designing'
    ]
    
    input_features = [
        skills_instance.database_fundamentals,
        skills_instance.computer_architecture,
        skills_instance.distributed_computing_systems,
        skills_instance.cyber_security,
        skills_instance.networking,
        skills_instance.software_development,
        skills_instance.programming_skills,
        skills_instance.project_management,
        skills_instance.computer_forensics_fundamentals,
        skills_instance.technical_communication,
        skills_instance.ai_ml,
        skills_instance.software_engineering,
        skills_instance.business_analysis,
        skills_instance.communication_skills,
        skills_instance.data_science,
        skills_instance.troubleshooting_skills,
        skills_instance.graphics_designing
    ]
    # print(input_features)
    # Create DataFrame with feature names
    X = pd.DataFrame([input_features], columns=feature_names)
    
    # Get prediction probabilities
    predictions = model.predict_proba(X)[0]
    
    # Create a sorted list of roles by probability
    role_probabilities = [
        (le.inverse_transform([i])[0], prob) 
        for i, prob in enumerate(predictions)
    ]

    
    # Sort roles by probability in descending order
    sorted_roles = sorted(role_probabilities, key=lambda x: x[1], reverse=True)
    # Filter and select top roles
    significant_roles = [
        role for role, prob in sorted_roles 
        if prob > 0.05  # Lowered threshold to capture more relevant roles
    ][:4]
    
    return significant_roles

def investigate_model():
    # Load the trained model and data
    data = pd.read_csv(r'C:\Users\vinod\external projects\career guidance website\career_advisor\dataset9000.csv')
    processed_data, label_encoder = preprocess_data(data)
    
    # Split features and target
    X = processed_data.drop('Role', axis=1)
    y = processed_data['Role']
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model
    model = create_model(X_train, y_train)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Class distribution
    class_distribution = y.value_counts(normalize=True)
    print("\nClass Distribution:")
    print(class_distribution)
    
    # Confusion matrix and classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



# Main execution for training the model
import os

def main():
    data = pd.read_csv(r'C:\Users\vinod\external projects\career guidance website\career_advisor\dataset9000.csv')
    
    processed_data, label_encoder = preprocess_data(data)
    
    # Split features and target
    X = processed_data.drop('Role', axis=1)
    y = processed_data['Role']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model directory if it doesn't exist
    os.makedirs('advisor/ml_models', exist_ok=True)
    
    # Create and train the model
    model = create_model(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save the trained model and label encoder
    with open('advisor/ml_models/career_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('advisor/ml_models/label_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)
    
    print("\nModel and Label Encoder saved successfully.")
    
    return model, label_encoder

if __name__ == "__main__":
    model, le = main()
    # investigate_model()
    
