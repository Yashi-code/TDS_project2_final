import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind

# Set up the OpenAI API token (replace with your actual token)
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if AIPROXY_TOKEN is None:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

# Set the proxy URL for OpenAI API
openai.api_base = "https://aiproxy.sanand.workers.dev/openai"
openai.api_key = AIPROXY_TOKEN

def analyze_data(filename):
    """Loads and analyzes the given CSV dataset.
    
    Args:
        filename (str): Path to the CSV file.
        
    Returns:
        tuple: A tuple containing summary statistics, missing values,
               correlation matrix, outliers, and trends.
    """
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {filename}. Trying with 'latin1' encoding.")
        df = pd.read_csv(filename, encoding='latin1')
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None, None, None

    numeric_df = df.select_dtypes(include='number')
    
    # Summary statistics
    summary_stats = numeric_df.describe().to_string()
    
    # Missing values
    missing_values = df.isnull().sum().to_string()
    
    # Correlation matrix
    correlation_matrix = numeric_df.corr() if not numeric_df.empty else None
    
    # Detecting outliers using IQR (Interquartile Range)
    outliers = detect_outliers(numeric_df)
    
    # Analyzing trends using linear regression
    trends = analyze_trends(df)
    
    return summary_stats, missing_values, correlation_matrix, outliers, trends, df

def detect_outliers(numeric_df):
    """Detects outliers in the numeric dataframe using IQR (Interquartile Range)."""
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    return outliers

def analyze_trends(df):
    """Analyzes trends in numeric data using linear regression."""
    numeric_df = df.select_dtypes(include='number')
    trend_results = {}
    
    if 'Time' in numeric_df.columns:
        X = numeric_df[['Time']]
        for column in numeric_df.columns:
            if column != 'Time':
                y = numeric_df[column]
                model = LinearRegression()
                model.fit(X, y)
                trend_results[column] = model.coef_[0]
    
    return trend_results

def visualize_data(df, dataset_name):
    """Generates key visualizations for the dataset."""
    sns.pairplot(df.select_dtypes(include='number'))
    plt.savefig(f'{dataset_name}/pairplot.png')
    plt.close()

    for column in df.select_dtypes(include='number').columns:
        plt.figure()
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.savefig(f'{dataset_name}/{column}_distribution.png')
        plt.close()

def perform_hypothesis_testing(df, column_pairs):
    """Performs t-tests for given column pairs."""
    results = {}
    for col1, col2 in column_pairs:
        stat, p_value = ttest_ind(df[col1].dropna(), df[col2].dropna())
        results[(col1, col2)] = p_value
    return results

def detect_anomalies(df):
    """Uses Isolation Forest to detect anomalies."""
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    numeric_df = df.select_dtypes(include='number')
    numeric_df['anomaly'] = model.fit_predict(numeric_df)
    anomalies = numeric_df[numeric_df['anomaly'] == -1]
    return anomalies

def create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, hypothesis_results, anomalies, dataset_description):
    """Uses LLM to create a narrative about the analysis."""
    correlation_matrix_markdown = correlation_matrix.to_markdown() if correlation_matrix is not None else "No correlation matrix available."
    
    prompt = f"""
Dataset Description: {dataset_description}
**Summary Statistics:** {summary_stats}
**Missing Values:** {missing_values}
**Correlation Matrix:** {correlation_matrix_markdown}
**Outliers:** {outliers}
**Trends (Regression Coefficients):** {trends}
**Hypothesis Test Results:** {hypothesis_results}
**Anomalies Detected:** {anomalies}

Create a structured narrative about the analysis results.
"""
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error: Failed to create story using LLM."

def create_folder(dataset_name):
    """Creates a folder for each dataset to store analysis files."""
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

def main(dataset_filenames):
    """Main function to run the analysis and create narratives for multiple datasets."""
    for dataset_filename in dataset_filenames:
        dataset_name = dataset_filename.split('.')[0]
        print(f"Analyzing {dataset_filename}...")
        
        # Create folder for each dataset
        create_folder(dataset_name)
        
        summary_stats, missing_values, correlation_matrix, outliers, trends, df = analyze_data(dataset_filename)
        
        # Visualize data
        visualize_data(df, dataset_name)
        
        # Perform hypothesis testing (example pairs)
        column_pairs = [('column1', 'column2'), ('column3', 'column4')]  # Modify pairs as needed
        hypothesis_results = perform_hypothesis_testing(df, column_pairs)
        
        # Detect anomalies
        anomalies = detect_anomalies(df)
        
        # Brief description of the dataset
        dataset_description = f"This dataset contains data about {dataset_name}."
        
        # Generate the story
        story = create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, hypothesis_results, anomalies, dataset_description)

        # Save the story to README.md
        with open(f'{dataset_name}/README.md', 'w') as f:
            f.write("# Automated Data Analysis\n")
            f.write(f"## Analysis of {dataset_filename}\n")
            f.write(f"### Summary Statistics\n{summary_stats}\n")
            f.write(f"### Missing Values\n{missing_values}\n")
            f.write(f"### Correlation Matrix\n{correlation_matrix}\n")
            f.write(f"### Outliers\n{outliers}\n")
            f.write(f"### Trend Analysis\n{trends}\n")
            f.write(f"### Hypothesis Test Results\n{hypothesis_results}\n")
            f.write(f"### Anomalies Detected\n{anomalies}\n")
            f.write(f"### Analysis Story\n{story}\n")
        
        print(f"Analysis for {dataset_filename} complete.\n")

if __name__ == "__main__":
    # List of datasets to process
    dataset_files = ['goodreads.csv', 'happiness.csv', 'media.csv']  # Modify as needed
    main(dataset_files)
