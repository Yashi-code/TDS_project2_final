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
    
    if not numeric_df.empty:
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        numeric_df['anomaly'] = model.fit_predict(numeric_df)
        return numeric_df[numeric_df['anomaly'] == -1]
    
    return pd.DataFrame()  # Return empty DataFrame if no numerical data

def perform_hypothesis_testing(df, column_pairs):
    """Performs t-tests for specified pairs of columns."""
    results = {}
    
    for col1, col2 in column_pairs:
        if col1 in df.columns and col2 in df.columns:
            stat, p_value = ttest_ind(df[col1].dropna(), df[col2].dropna())
            results[(col1, col2)] = p_value
            
    return results

# ========== 3. Visualization ==========
def visualize_data(df, dataset_name):
    """Generates visualizations for the dataset with titles, axis labels, and legends."""
    
    # Pairplot for all numerical features
    sns.pairplot(df.select_dtypes(include='number'))
    plt.title(f"Pairplot of {dataset_name}")
    plt.savefig(f'{dataset_name}/pairplot.png')
    
    # Histograms for each numerical feature
    for column in df.select_dtypes(include='number').columns:
        plt.figure()
        sns.histplot(df[column], kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend([f"Distribution of {column}"])
        plt.savefig(f'{dataset_name}/{column}_distribution.png')
        
# ========== 4. Narrative ==========
def create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, hypothesis_results, anomalies, dataset_description):
    """Creates a context-rich narrative summary for the analysis."""
    
    correlation_matrix_md = correlation_matrix.to_markdown() if correlation_matrix is not None else "No correlation matrix available."
    
    anomaly_str = anomalies.to_string() if not anomalies.empty else "No anomalies detected."
    
    prompt = f""" 
Dataset Description: {dataset_description} 
**Summary Statistics:** {summary_stats} 
**Missing Values:** {missing_values} 
**Correlation Matrix:** {correlation_matrix_md} 
**Outliers:** {outliers} 
**Trends (Regression Coefficients):** {trends} 
**Hypothesis Test Results:** {hypothesis_results} 
**Anomalies Detected:** {anomaly_str} 

Create a structured narrative summary of this data analysis with the following: 
1. Briefly describe the dataset.
2. Explain the data analysis and key insights.
3. Highlight surprising or significant findings.
4. Discuss implications and suggested actions based on significant findings.
5. Ensure proper Markdown formatting for easy readability.
6. Integrate visualizations at relevant points and emphasize significant findings.
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

# ========== 5. Efficient LLM Usage ==========
def efficient_llm_usage(data):
    """Minimize token usage by sending concise prompts to LLM."""
    
    # Here we only send relevant insights and summaries instead of large datasets.
    
    summary = data['summary_stats']
    
    insights = f"Key insights from the analysis: {summary}"
    
    return insights

# ========== 6. Dynamic Prompts and Function Calling ==========
def generate_dynamic_prompt(data):
    """Generates a dynamic prompt for LLM based on the dataset's specific features."""
    
    prompt = f"Analyze the dataset with the following properties:\n{data['features']}\nProvide insights into trends, correlations, and anomalies."
    
    return prompt

def dynamic_function_call(data, function_type="analysis"):
    """Dynamically call the appropriate function based on the data type."""
    
    if function_type == "analysis":
        return analyze_trends(data)
    
    elif function_type == "visualization":
        visualize_data(data, "dynamic_dataset")  # Visualization doesn't need a return value
    
    elif function_type == "narrative":
        return create_story(data['summary_stats'], data['missing_values'], data['correlation_matrix'], data['outliers'], data['trends'], data['hypothesis_results'], data['anomalies'], "Sample Dataset")
    
    else:
        return "Invalid function type."

# ========== 7. Vision Agentic (Vision + Multiple LLM Calls) ==========
def vision_agentic_workflow(df, dataset_name):
   """Vision-based agentic workflow with multiple LLM calls."""
   
   visualize_data(df, dataset_name)  # First generate visualizations
   
   # Get summary and analysis
   summary_stats = get_summary_stats(df)
   missing_values = detect_missing_values(df)
   correlation_matrix = calculate_correlation_matrix(df)
   outliers = detect_outliers(df)
   trends = analyze_trends(df)
   anomalies = detect_anomalies(df)
   
   column_pairs = [('column1', 'column2'), ('column3', 'column4')]  # Update as needed
   hypothesis_results = perform_hypothesis_testing(df, column_pairs)

   # Prepare the data for LLM processing
   analysis_data = {
       'summary_stats': summary_stats,
       'features': "Key numerical features of the dataset",
       'trends': trends,
       'missing_values': missing_values,
       'correlation_matrix': correlation_matrix,
       'outliers': outliers,
       'hypothesis_results': hypothesis_results,
       'anomalies': anomalies,
   }
   
   prompt = generate_dynamic_prompt(analysis_data)  # Generate dynamic prompt
   
   insights = efficient_llm_usage(analysis_data)  # Generate concise insights
   
   # Call LLM for final narrative
   narrative = create_story(
       summary_stats,
       missing_values,
       correlation_matrix,
       outliers,
       trends,
       hypothesis_results,
       anomalies,
       "Dataset description"
   )
   
   return insights, narrative

# ========== 8. Main Execution ==========
def analyze_dataset(dataset_filename):
     """Performs end-to-end analysis for a single dataset."""
     
     dataset_name = dataset_filename.split('.')[0]
     print(f"Analyzing {dataset_filename}...")
     create_folder(dataset_name)
     df = load_data(dataset_filename)
     
     if df is None:
         return

     # Dynamic function calling based on data
     dynamic_function_call(df, "analysis")
     dynamic_function_call(df, "visualization")
     dynamic_function_call(df, "narrative")

     # Vision Agentic Workflow
     insights, narrative = vision_agentic_workflow(df, dataset_name)

     print(f"Insights: {insights}")
     print(f"Narrative: {narrative}")
     
     print(f"Analysis for {dataset_filename} complete.\n")

if __name__ == "__main__":
     dataset_files = ['goodreads.csv', 'happiness.csv', 'media.csv']
     for dataset in dataset_files:
         analyze_dataset(dataset)
