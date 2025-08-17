#!/usr/bin/env python3
"""
Titanic Survival Predictor - Command Line Interface
A simplified version of the machine learning model for quick predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class TitanicPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, filepath='Titanic-Dataset.csv'):
        """Load and preprocess the Titanic dataset"""
        print("📊 Loading Titanic dataset...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Feature engineering
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
        df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
        
        # Extract titles
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df['Title_Grouped'] = df['Title'].map(title_mapping).fillna('Rare')
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], 
                                labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
        
        # Fare groups
        df['Fare_Group'] = pd.cut(df['Fare'], bins=[0, 8, 15, 31, 100, 1000], 
                                 labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        # Handle missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Recalculate categorical features after filling missing values
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], 
                                labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
        df['Fare_Group'] = pd.cut(df['Fare'], bins=[0, 8, 15, 31, 100, 1000], 
                                 labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        return df
    
    def train_model(self, df):
        """Train the survival prediction model"""
        print("🤖 Training machine learning model...")
        
        # Select features
        feature_columns = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Family_Size', 'Is_Alone', 'Title_Grouped', 'Age_Group', 'Fare_Group'
        ]
        
        X = df[feature_columns].copy()
        y = df['Survived']
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Embarked', 'Title_Grouped', 'Age_Group', 'Fare_Group']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            self.encoders[feature] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, 
            min_samples_leaf=2, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"✅ Training accuracy: {train_accuracy:.4f}")
        print(f"✅ Test accuracy: {test_accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        self.feature_columns = feature_columns
        return test_accuracy
    
    def predict_survival(self, pclass, sex, age, sibsp, parch, fare, embarked):
        """Predict survival for a given passenger profile"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Create input dataframe
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Determine title based on age and sex
        if sex.lower() == 'male':
            title = 'Master' if age < 18 else 'Mr'
        else:
            title = 'Miss'  # Simplified for this demo
        
        # Create age group
        if age <= 12:
            age_group = 'Child'
        elif age <= 18:
            age_group = 'Teen'
        elif age <= 30:
            age_group = 'Young_Adult'
        elif age <= 50:
            age_group = 'Adult'
        else:
            age_group = 'Senior'
        
        # Create fare group
        if fare <= 8:
            fare_group = 'Very_Low'
        elif fare <= 15:
            fare_group = 'Low'
        elif fare <= 31:
            fare_group = 'Medium'
        elif fare <= 100:
            fare_group = 'High'
        else:
            fare_group = 'Very_High'
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex.lower()],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked.upper()],
            'Family_Size': [family_size],
            'Is_Alone': [is_alone],
            'Title_Grouped': [title],
            'Age_Group': [age_group],
            'Fare_Group': [fare_group]
        })
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Embarked', 'Title_Grouped', 'Age_Group', 'Fare_Group']
        
        for feature in categorical_features:
            if feature in self.encoders:
                value = input_data[feature].iloc[0]
                if value in self.encoders[feature].classes_:
                    input_data[feature] = self.encoders[feature].transform([value])[0]
                else:
                    # Use most common class for unseen categories
                    input_data[feature] = 0
        
        # Make prediction
        survival_probability = self.model.predict_proba(input_data)[0, 1]
        survival_prediction = self.model.predict(input_data)[0]
        
        return survival_probability, survival_prediction

def interactive_prediction():
    """Run interactive prediction interface"""
    print("\n" + "="*60)
    print("🚢 TITANIC SURVIVAL PREDICTOR")
    print("="*60)
    print("Enter your information to see if you would have survived!")
    print()
    
    try:
        # Get user input
        print("🎫 Passenger Class:")
        print("   1 = First Class (luxury)")
        print("   2 = Second Class (middle)")
        print("   3 = Third Class (economy)")
        pclass = int(input("Enter your class (1-3): "))
        if pclass not in [1, 2, 3]:
            pclass = 3
            print("   Using 3rd class")
        
        print("\n👤 Gender:")
        sex = input("Enter your gender (male/female): ").lower().strip()
        if sex not in ['male', 'female']:
            sex = 'male'
            print("   Using 'male'")
        
        print("\n🎂 Age:")
        age = float(input("Enter your age: "))
        if age < 0 or age > 120:
            age = 30
            print("   Using age 30")
        
        print("\n👨‍👩‍👧‍👦 Family Information:")
        sibsp = int(input("Number of siblings/spouses aboard: "))
        if sibsp < 0:
            sibsp = 0
        
        parch = int(input("Number of parents/children aboard: "))
        if parch < 0:
            parch = 0
        
        print("\n💰 Fare:")
        print("   Typical: 1st class (~$80), 2nd class (~$20), 3rd class (~$15)")
        fare = float(input("Enter fare (1912 dollars): "))
        if fare < 0:
            fare = 15
            print("   Using $15")
        
        print("\n🚢 Embarkation Port:")
        print("   S = Southampton, C = Cherbourg, Q = Queenstown")
        embarked = input("Enter port (S/C/Q): ").upper().strip()
        if embarked not in ['S', 'C', 'Q']:
            embarked = 'S'
            print("   Using Southampton")
        
        # Make prediction
        print("\n🔮 Analyzing your survival chances...")
        prob, pred = predictor.predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULTS")
        print("="*60)
        
        print(f"\n👤 Your Profile:")
        print(f"   Class: {pclass} ({'First' if pclass==1 else 'Second' if pclass==2 else 'Third'})")
        print(f"   Gender: {sex.title()}")
        print(f"   Age: {age} years")
        print(f"   Family: {sibsp + parch} relatives aboard")
        print(f"   Fare: ${fare:.2f}")
        print(f"   Port: {embarked}")
        
        print(f"\n🎯 Prediction:")
        if pred == 1:
            print(f"   ✅ YOU WOULD HAVE SURVIVED! 🎉")
        else:
            print(f"   ❌ You would not have survived 😢")
        
        print(f"   Survival Probability: {prob:.1%}")
        
        # Provide insights
        print(f"\n💡 Key Factors:")
        if sex == 'female':
            print(f"   + Being female greatly increased survival (74% vs 19%)")
        else:
            print(f"   - Being male decreased survival chances")
        
        if pclass == 1:
            print(f"   + First class had highest survival rate (63%)")
        elif pclass == 2:
            print(f"   ○ Second class had moderate survival (47%)")
        else:
            print(f"   - Third class had lowest survival rate (24%)")
        
        if age < 18:
            print(f"   + Children had better survival chances")
        
        family_size = sibsp + parch + 1
        if family_size == 1:
            print(f"   - Traveling alone reduced chances")
        elif 2 <= family_size <= 4:
            print(f"   + Small family size helped")
        else:
            print(f"   - Large families had coordination difficulties")
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Error: {e}")
        print("Please run again with valid inputs.")

def main():
    """Main function"""
    global predictor
    
    print("🚢 Titanic Survival Predictor")
    print("=" * 40)
    
    # Initialize predictor
    predictor = TitanicPredictor()
    
    # Load and train model
    df = predictor.load_and_prepare_data()
    accuracy = predictor.train_model(df)
    
    print(f"\n🎉 Model ready! (Accuracy: {accuracy:.1%})")
    
    # Show some example predictions
    print("\n📊 Example Predictions:")
    examples = [
        (1, 'female', 25, 0, 0, 80, 'S', "Wealthy young woman"),
        (3, 'male', 30, 1, 1, 15, 'S', "Working class father"),
        (2, 'female', 35, 0, 2, 25, 'C', "Middle class mother"),
        (3, 'male', 22, 0, 0, 7, 'Q', "Poor young man alone")
    ]
    
    for pclass, sex, age, sibsp, parch, fare, embarked, desc in examples:
        prob, pred = predictor.predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
        result = "✅ Survived" if pred == 1 else "❌ Did not survive"
        print(f"  {desc}: {result} ({prob:.1%})")
    
    # Run interactive interface
    while True:
        print("\n" + "="*40)
        choice = input("Try the predictor? (y/n/q to quit): ").lower().strip()
        if choice in ['q', 'quit', 'exit']:
            break
        elif choice in ['y', 'yes', '']:
            interactive_prediction()
        elif choice in ['n', 'no']:
            print("Thanks for using the Titanic Predictor!")
            break

if __name__ == "__main__":
    main()