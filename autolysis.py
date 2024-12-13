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

openai.api_base = "https://aiproxy.sanand.workers.dev/openai"
openai.api_key = AIPROXY_TOKEN


# ========== 1. Data Handling ==========
def load_data(filename):
    """Loads data from a CSV file, handling common encoding issues."""
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {filename}. Trying with 'latin1' encoding.")
        df = pd.read_csv(filename, encoding='latin1')
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    return df


def create_folder(dataset_name):
    """Creates a folder for storing analysis outputs."""
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)


# ========== 2. Analysis ==========
def get_summary_stats(df):
    """Generates summary statistics for numerical data."""
    numeric_df = df.select_dtypes(include='number')
    return numeric_df.describe().to_string()


def detect_missing_values(df):
    """Counts missing values in the dataset."""
    return df.isnull().sum().to_string()


def calculate_correlation_matrix(df):
    """Calculates correlation matrix for numerical columns."""
    numeric_df = df.select_dtypes(include='number')
    return numeric_df.corr() if not numeric_df.empty else None


def detect_outliers(df):
    """Detects outliers in numerical columns using the IQR method."""
    numeric_df = df.select_dtypes(include='number')
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    return ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()


def analyze_trends(df):
    """Analyzes trends in numerical data using linear regression."""
    numeric_df = df.select_dtypes(include='number')
    trend_results = {}

    if 'Time' in numeric_df.columns:
        X = numeric_df[['Time']]
        for column in numeric_df.columns:
            if column != 'Time':
                y = numeric_df[column]
                model = LinearRegression()
                model.fit(X, y)
                trend_results[column] = model.coef_[0]  # Coefficient of the regression line

    return trend_results


def detect_anomalies(df):
    """Detects anomalies using the Isolation Forest algorithm."""
    numeric_df = df.select_dtypes(include='number')
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    numeric_df['anomaly'] = model.fit_predict(numeric_df)
    return numeric_df[numeric_df['anomaly'] == -1]


def perform_hypothesis_testing(df, column_pairs):
    """Performs t-tests for specified pairs of columns."""
    results = {}
    for col1, col2 in column_pairs:
        stat, p_value = ttest_ind(df[col1].dropna(), df[col2].dropna())
        results[(col1, col2)] = p_value
    return results


# ========== 3. Visualization ==========
def visualize_data(df, dataset_name):
    """Generates visualizations for the dataset."""
    sns.pairplot(df.select_dtypes(include='number'))
    plt.savefig(f'{dataset_name}/pairplot.png')
    plt.close()

    for column in df.select_dtypes(include='number').columns:
        plt.figure()
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.savefig(f'{dataset_name}/{column}_distribution.png')
        plt.close()


# ========== 4. Story Creation ==========
def create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, hypothesis_results, anomalies, dataset_description):
    """Creates a narrative summary using OpenAI's API."""
    correlation_matrix_md = correlation_matrix.to_markdown() if correlation_matrix is not None else "No correlation matrix available."

    prompt = f"""
Dataset Description: {dataset_description}
**Summary Statistics:** {summary_stats}
**Missing Values:** {missing_values}
**Correlation Matrix:** {correlation_matrix_md}
**Outliers:** {outliers}
**Trends (Regression Coefficients):** {trends}
**Hypothesis Test Results:** {hypothesis_results}
**Anomalies Detected:** {anomalies}

Create a structured narrative summary of this data analysis.
"""
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error: Unable to generate story."


# ========== 5. Main Execution ==========
def analyze_dataset(dataset_filename):
    """Performs end-to-end analysis for a single dataset."""
    dataset_name = dataset_filename.split('.')[0]
    print(f"Analyzing {dataset_filename}...")

    create_folder(dataset_name)
    df = load_data(dataset_filename)
    if df is None:
        return

    summary_stats = get_summary_stats(df)
    missing_values = detect_missing_values(df)
    correlation_matrix = calculate_correlation_matrix(df)
    outliers = detect_outliers(df)
    trends = analyze_trends(df)
    anomalies = detect_anomalies(df)

    column_pairs = [('column1', 'column2'), ('column3', 'column4')]  # Update as needed
    hypothesis_results = perform_hypothesis_testing(df, column_pairs)

    visualize_data(df, dataset_name)

    dataset_description = f"This dataset contains data about {dataset_name}."
    story = create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, hypothesis_results, anomalies, dataset_description)

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
        f.write(f"### Narrative Summary\n{story}\n")

    print(f"Analysis for {dataset_filename} complete.\n")


if __name__ == "__main__":
    dataset_files = ['goodreads.csv', 'happiness.csv', 'media.csv']
    for dataset in dataset_files:
        analyze_dataset(dataset)
