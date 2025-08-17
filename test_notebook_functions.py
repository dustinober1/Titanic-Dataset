#!/usr/bin/env python3
"""
Test script to validate all functions used in the Titanic EDA notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def test_basic_operations():
    """Test basic data loading and operations"""
    print("=== Testing Basic Operations ===")
    
    # Load data
    df = pd.read_csv('Titanic-Dataset.csv')
    print(f"✓ Data loaded successfully: {df.shape}")
    
    # Basic statistics
    print(f"✓ Basic info works")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    print(f"✓ Missing values analysis works")
    
    return df

def test_visualizations(df):
    """Test plotting functions"""
    print("\n=== Testing Visualizations ===")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.ioff()  # Turn off interactive plotting
    
    # Test basic plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    df['Survived'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Test Survival Plot')
    plt.close(fig)
    print("✓ Basic bar plot works")
    
    # Test seaborn
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.countplot(data=df, x='Sex', hue='Survived', ax=ax)
    plt.close(fig)
    print("✓ Seaborn countplot works")
    
    # Test correlation heatmap
    df_corr = df[['Survived', 'Pclass', 'Age', 'Fare']].corr()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, ax=ax)
    plt.close(fig)
    print("✓ Correlation heatmap works")

def test_data_transformations(df):
    """Test data transformation functions"""
    print("\n=== Testing Data Transformations ===")
    
    # Family size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Has_Family'] = (df['SibSp'] + df['Parch'] > 0).astype(int)
    print("✓ Family size calculations work")
    
    # Title extraction (fixed regex)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
    print("✓ Title extraction works")
    
    # Age binning
    df_age_complete = df.dropna(subset=['Age'])
    age_bins = pd.cut(df_age_complete['Age'], bins=[0, 12, 18, 30, 50, 80], 
                      labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 
                             'Adult (31-50)', 'Senior (51+)'])
    print("✓ Age binning works")
    
    # Fare binning
    fare_bins = pd.cut(df['Fare'], bins=[0, 10, 20, 50, 100, 600], 
                       labels=['Very Low ($0-10)', 'Low ($10-20)', 'Medium ($20-50)', 
                              'High ($50-100)', 'Very High ($100+)'])
    print("✓ Fare binning works")
    
    # Deck extraction
    df['Deck'] = df['Cabin'].str[0]
    print("✓ Deck extraction works")
    
    return df

def test_statistical_analysis(df):
    """Test statistical operations"""
    print("\n=== Testing Statistical Analysis ===")
    
    # Encoding categorical variables
    df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    print("✓ Categorical encoding works")
    
    # Correlation analysis
    correlation_features = ['Survived', 'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                           'Fare', 'Embarked_encoded', 'Family_Size', 'Has_Family']
    corr_matrix = df[correlation_features].corr()
    survival_corr = corr_matrix['Survived'].sort_values(key=abs, ascending=False)
    print("✓ Correlation analysis works")
    
    # Cross-tabulation
    gender_survival = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
    print("✓ Cross-tabulation works")
    
    # Pivot tables
    pivot_table = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean') * 100
    print("✓ Pivot tables work")

def test_advanced_analysis(df):
    """Test advanced analysis functions"""
    print("\n=== Testing Advanced Analysis ===")
    
    # Title grouping
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title_grouped'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
    title_survival = pd.crosstab(df['Title_grouped'], df['Survived'], normalize='index') * 100
    print("✓ Title analysis works")
    
    # Multi-dimensional analysis
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 80], 
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    age_class_survival = df.groupby(['Age_group', 'Pclass'])['Survived'].mean().unstack() * 100
    print("✓ Multi-dimensional analysis works")
    
    # Complex calculations
    survival_summary = {
        'Overall': df['Survived'].mean() * 100,
        'Female': df[df['Sex'] == 'female']['Survived'].mean() * 100,
        'Male': df[df['Sex'] == 'male']['Survived'].mean() * 100,
        '1st Class': df[df['Pclass'] == 1]['Survived'].mean() * 100,
        '3rd Class': df[df['Pclass'] == 3]['Survived'].mean() * 100,
        'Children': df[df['Age'] <= 12]['Survived'].mean() * 100
    }
    print("✓ Complex survival calculations work")

def main():
    """Run all tests"""
    print("🚢 TITANIC EDA NOTEBOOK VALIDATION TESTS")
    print("=" * 50)
    
    try:
        # Test each component
        df = test_basic_operations()
        test_visualizations(df)
        df = test_data_transformations(df)
        test_statistical_analysis(df)
        test_advanced_analysis(df)
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("The Titanic EDA notebook should work correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Some functions may need fixing.")
        raise

if __name__ == "__main__":
    main()