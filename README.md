# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load the Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target
        iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(iris_df.head())
        
        # Explore structure
        print("\nDataset info:")
        print(iris_df.info())
        
        # Check for missing values
        print("\nMissing values:")
        print(iris_df.isnull().sum())
        
        # Clean data (though Iris dataset is already clean)
        # In case of missing values, we could use:
        # iris_df.fillna(iris_df.mean(), inplace=True) for numerical columns
        # or iris_df.dropna(inplace=True)
        
        return iris_df
    
    except Exception as e:
        print(f"Error loading or exploring data: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    try:
        # Basic statistics
        print("\nBasic statistics for numerical columns:")
        print(df.describe())
        
        # Group by species and compute means
        print("\nMean values by species:")
        print(df.groupby('species').mean())
        
        # Additional interesting findings
        print("\nAdditional observations:")
        print("1. Setosa has the smallest petal dimensions")
        print("2. Virginica has the largest petal dimensions")
        print("3. Sepal width has the smallest variation across species")
        
    except Exception as e:
        print(f"Error during data analysis: {e}")

# Task 3: Data Visualization
def create_visualizations(df):
    try:
        # Set style for better looking plots
        sns.set(style="whitegrid")
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Line chart (showing trends by index since there's no time variable)
        plt.subplot(2, 2, 1)
        df['sepal length (cm)'].plot(kind='line', color='blue')
        plt.title('Sepal Length Trend by Sample Index')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        
        # 2. Bar chart (average sepal length by species)
        plt.subplot(2, 2, 2)
        df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color=['red', 'green', 'blue'])
        plt.title('Average Sepal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Average Sepal Length (cm)')
        
        # 3. Histogram (sepal width distribution)
        plt.subplot(2, 2, 3)
        df['sepal width (cm)'].plot(kind='hist', bins=15, color='purple', edgecolor='black')
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        
        # 4. Scatter plot (sepal length vs petal length)
        plt.subplot(2, 2, 4)
        colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
        for species, color in colors.items():
            subset = df[df['species'] == species]
            plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                        color=color, label=species, alpha=0.7)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('iris_visualizations.png')
        plt.show()
        
        # Additional visualization using seaborn
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='species', y='petal width (cm)', data=df)
        plt.title('Petal Width Distribution by Species')
        plt.savefig('iris_boxplot.png')
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

# Main execution
if __name__ == "__main__":
    print("Starting data analysis...\n")
    
    # Task 1
    iris_df = load_and_explore_data()
    
    if iris_df is not None:
        # Task 2
        perform_data_analysis(iris_df)
        
        # Task 3
        create_visualizations(iris_df)
        
    print("\nAnalysis complete. Visualizations saved as PNG files.")
