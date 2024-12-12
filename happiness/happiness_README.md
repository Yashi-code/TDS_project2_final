# Automated Data Analysis
## Analysis of happiness.csv
### Summary Statistics
              year  Life Ladder  Log GDP per capita  Social support  Healthy life expectancy at birth  Freedom to make life choices   Generosity  Perceptions of corruption  Positive affect  Negative affect
count  2363.000000  2363.000000         2335.000000     2350.000000                       2300.000000                   2327.000000  2282.000000                2238.000000      2339.000000      2347.000000
mean   2014.763860     5.483566            9.399671        0.809369                         63.401828                      0.750282     0.000098                   0.743971         0.651882         0.273151
std       5.059436     1.125522            1.152069        0.121212                          6.842644                      0.139357     0.161388                   0.184865         0.106240         0.087131
min    2005.000000     1.281000            5.527000        0.228000                          6.720000                      0.228000    -0.340000                   0.035000         0.179000         0.083000
25%    2011.000000     4.647000            8.506500        0.744000                         59.195000                      0.661000    -0.112000                   0.687000         0.572000         0.209000
50%    2015.000000     5.449000            9.503000        0.834500                         65.100000                      0.771000    -0.022000                   0.798500         0.663000         0.262000
75%    2019.000000     6.323500           10.392500        0.904000                         68.552500                      0.862000     0.093750                   0.867750         0.737000         0.326000
max    2023.000000     8.019000           11.676000        0.987000                         74.600000                      0.985000     0.700000                   0.983000         0.884000         0.705000
### Missing Values
Country name                          0
year                                  0
Life Ladder                           0
Log GDP per capita                   28
Social support                       13
Healthy life expectancy at birth     63
Freedom to make life choices         36
Generosity                           81
Perceptions of corruption           125
Positive affect                      24
Negative affect                      16
### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)
### Outliers
![Outliers](outliers.png)
### Trend Analysis
![Trends](trends.png)
### Analysis Story
### Data Narrative and Analysis Summary

This analysis focuses on a dataset of 2,363 observations related to various well-being metrics across countries, capturing data on subjective well-being (Life Ladder), economic indicators (Log GDP per capita), social indicators (Social support, Healthy life expectancy at birth), individual freedoms, generosity, perceptions of corruption, and emotional states (Positive and Negative affect).

#### 1. Summary Statistics
The summary statistics reveal the following insights:

- **Life Ladder**: The average life satisfaction score is approximately 5.48 (on a scale of 0-10), with a standard deviation of 1.13, indicating variability in life satisfaction across the countries studied.
- **Log GDP per capita**: The mean GDP per capita (on a logarithmic scale) is around 9.40, with a range suggesting significant economic diversity among countries.
- **Social Support**: The mean support score is 0.81, showcasing generally high levels of social support.
- **Healthy Life Expectancy**: Averaging 63.40 years, the data highlights differences in health outcomes across regions.
- **Freedom to Make Life Choices**: This average score is relatively high at 0.75, indicating that many individuals perceive themselves as having the freedom to make their choices in life.
- **Generosity and Corruption Perceptions**: Both metrics are on a low scale, with average generosity near zero and perceptions of corruption reflecting a moderate level of concern.

#### 2. Missing Values
The dataset exhibits several missing values, primarily in the following variables:
- **Generosity (81 missing)** and **Perceptions of Corruption (125 missing)** have the highest counts of missing entries, suggesting potential issues in data collection or reporting.
- Other variables like **Log GDP per capita**, **Social Support**, and **Healthy Life Expectancy** also show missing values, but to a lesser extent.

These missing values may introduce biases in analysis and need addressing, possibly through imputation techniques or sensitivity analysis.

#### 3. Correlation Matrix
A correlation matrix analysis indicates the following significant relationships:

- **Life Ladder and Log GDP per capita**: A strong positive correlation (r = 0.78) suggests that higher economic prosperity correlates with greater life satisfaction.
- **Social Support**: It also correlates positively with Life Ladder (r = 0.72) and Log GDP per capita (r = 0.69), indicating that both social support and economic wealth significantly contribute to overall well-being.
- **Negative Affect and Positive Affect**: A notable negative correlation (r = -0.33) exists here, conveying that lower negative emotions align with higher positive feelings.
- **Freedom to Make Life Choices**: This variable shows a positive correlation with Life Ladder (r = 0.54), suggesting that perceived freedoms contribute to well-being.
- **Perceptions of Corruption**: A negative correlation with Life Ladder (r = -0.43) indicates that countries where corruption is perceived to be high tend to report lower life satisfaction.

#### 4. Outliers
The analysis identifies outliers in several key variables:
- Outliers are most notable in **Perceptions of Corruption** (194 outliers) and **Social Support** (48 outliers), highlighting potential anomalies in these variables.
- Outliers in **Life Ladder** (2 outliers) and **Log GDP per capita** (1 outlier) suggest extreme cases that could skew mean values and potentially affect regression analyses.

Identifying and understanding these outliers’ nature is crucial for interpreting results accurately.

#### 5. Trends
The regression coefficients section is currently empty. Conducting regression analyses could provide insights into the relationships between independent variables (e.g., Log GDP per capita, Social Support) and the dependent variable (Life Ladder). Such analysis would help in quantifying the impact of various predictors on life satisfaction.

### Key Findings and Implications
- **Economic prosperity** and **social support** are pivotal factors influencing life satisfaction, calling for policies that enhance economic growth and social cohesion.
- The significant **negative correlation between perceptions of corruption and life satisfaction** indicates the importance of good governance and transparency in improving citizen well-being.
- The presence of **outliers, especially in perceptions of corruption**, suggests that addressing these extreme cases might provide deeper insights into specific countries or regions' governance challenges.
- **Missing values** in key metrics like Generosity and Corruption Perceptions highlight the need for improved data collection efforts and potential biases in analysis.

In conclusion, this extensive analysis of the dataset opens avenues for further exploration into the interconnectedness of economic, social, and emotional variables influencing life satisfaction on a global scale. Addressing the noted gaps and outliers will enhance the robustness of any conclusions drawn and enable meaningful policy recommendations.
