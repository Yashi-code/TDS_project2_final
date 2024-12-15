# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "dask",
#   "ipykernel",
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from dask import dataframe as dd

def analyze_data(df):
    """
    Analyze the dataset to compute summary statistics, missing values, and correlations.
    
    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        tuple: summary_stats (DataFrame), missing_values (Series), corr_matrix (DataFrame).
    """
    print("Analyzing the data...")
    summary_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_values, corr_matrix

def detect_outliers(df):
    """
    Detect outliers in the dataset using the Isolation Forest method.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.Series: A series containing the count of anomalies per column.
    """
    print("Detecting outliers...")
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        return pd.Series(dtype=int)

    iso_forest = IsolationForest(random_state=42, contamination=0.05)
    df_numeric['anomaly'] = iso_forest.fit_predict(df_numeric)
    outliers = df_numeric['anomaly'].value_counts()
    print("Outliers detection complete.")
    return outliers

def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate visualizations: correlation heatmap, outlier bar plot, PCA plot, missing value heatmap, and data distribution plot.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts.
        df (pd.DataFrame): The input dataset.
        output_dir (str): Directory to save visualizations.

    Returns:
        list: Paths to the generated visualization files.
    """
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    files = []

    # Correlation Heatmap
    if not corr_matrix.empty:
        heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig(heatmap_file)
        files.append(heatmap_file)
        plt.close()

    # PCA Visualization
    pca_file = os.path.join(output_dir, 'pca_plot.png')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[numeric_columns].dropna())
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        plt.title('PCA Plot')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(pca_file)
        files.append(pca_file)
        plt.close()

    # Missing Value Heatmap
    missing_file = os.path.join(output_dir, 'missing_values.png')
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(missing_file)
    files.append(missing_file)
    plt.close()

    print("Visualizations generated.")
    return files

def create_readme(summary_stats, missing_values, corr_matrix, outliers, visual_files, output_dir):
    """
    Create a README.md file summarizing the analysis and visualizations.

    Args:
        summary_stats (pd.DataFrame): Summary statistics.
        missing_values (pd.Series): Missing values.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts.
        visual_files (list): Paths to visualization files.
        output_dir (str): Output directory.

    Returns:
        str: Path to the README file.
    """
    print("Creating README file...")
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")

        # Summary Statistics
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")

        # Missing Values
        f.write("## Missing Values\n")
        f.write(missing_values.to_string())
        f.write("\n\n")

        # Outliers
        if outliers.sum() > 0:
            f.write("## Outliers\n")
            f.write(outliers.to_string())
            f.write("\n\n")

        # Visualizations
        f.write("## Visualizations\n")
        for file in visual_files:
            f.write(f"![{os.path.basename(file)}]({file})\n\n")

    print("README file created.")
    return readme_file

def question_llm(prompt, context):
    """
    Generate a contextual story using OpenAI's proxy API.

    Args:
        prompt (str): Prompt for the LLM.
        context (str): Dataset summary to provide context.

    Returns:
        str: Generated story.
    """
    print("Generating narrative using LLM...")
    try:
        token = os.environ["AIPROXY_TOKEN"]
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data analyst generating narratives for datasets."},
                {"role": "user", "content": f"{prompt}\nContext: {context}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return "Failed to generate story."
    except Exception as e:
        return f"Error: {e}"

def main(csv_file):
    print("Starting analysis...")
    try:
        # Leverage Dask for large datasets
        if os.stat(csv_file).st_size > 1e7:  # If file size > 10MB
            df = dd.read_csv(csv_file).compute()
        else:
            df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    visual_files = visualize_data(corr_matrix, outliers, df, output_dir)
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, visual_files, output_dir)

    # Append story to README
    context = summary_stats.to_string()
    story = question_llm("Write a summary and findings based on the dataset analysis.", context)
    with open(readme_file, 'a') as f:
        f.write("\n## Narrative Summary\n")
        f.write(f"{story}\n")
    print(f"Analysis complete. Results saved in '{output_dir}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
