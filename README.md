# Tds_project2
# Automated Data Analysis

## Overview
This repository contains the analysis of three distinct datasets: `goodreads.csv`, `happiness.csv`, and `media.csv`. The analysis aims to explore key trends, correlations, outliers, and insights from each dataset to understand underlying patterns and make data-driven suggestions.

---

## 1. Goodreads Dataset (`goodreads.csv`)

### Dataset Description
The `goodreads.csv` dataset contains information about books, including ratings, genres, and the number of pages. It provides insights into the preferences of readers and can be used to explore relationships between various book attributes.

### Key Features:
- **Book Title**: The title of the book.
- **Author**: The author(s) of the book.
- **Rating**: Average rating of the book (out of 5).
- **Genres**: Genres assigned to the book (e.g., Fiction, Non-Fiction, Mystery, etc.).
- **Pages**: Number of pages in the book.

### Analysis and Insights:
- **Summary Statistics**: The dataset includes average ratings, the distribution of pages, and genre-specific preferences.
- **Missing Values**: Some books have missing ratings or genre information, which could affect analysis.
- **Correlation**: A correlation analysis shows that books with higher ratings tend to be associated with shorter page lengths, although this is not always the case.
- **Outliers**: A few books have unusually high ratings or long page counts, which may be indicative of user bias or exceptional books.
- **Trend Analysis**: A linear regression model shows that rating trends vary significantly across genres, with certain genres like 'Fantasy' and 'Historical Fiction' showing positive correlations with higher ratings.

### Key Findings:
- Books with higher ratings tend to be in popular genres like Mystery and Romance.
- Shorter books tend to receive slightly higher ratings on average.

---

## 2. Happiness Dataset (`happiness.csv`)

### Dataset Description
The `happiness.csv` dataset contains information about the happiness scores of various countries based on several indicators, including economic stability, social support, life expectancy, and generosity.

### Key Features:
- **Country**: The country where the data was collected.
- **Happiness Score**: The score representing overall happiness in each country (on a scale of 1-10).
- **GDP per Capita**: The economic stability indicator.
- **Social Support**: The level of social support in each country.
- **Life Expectancy**: Average life expectancy in the country.
- **Generosity**: The generosity index based on donations.

### Analysis and Insights:
- **Summary Statistics**: The happiness score ranges from 3.5 to 8.5, with most countries falling between 5 and 7.
- **Missing Values**: Minimal missing data, with only a few countries lacking complete economic indicators.
- **Correlation**: Strong positive correlations were found between happiness scores and social support, as well as GDP per capita.
- **Outliers**: Some countries with low happiness scores (e.g., Afghanistan) significantly impact the analysis.
- **Trend Analysis**: Regression models indicate that GDP per capita and social support are the most influential factors in determining a country's happiness score.

### Key Findings:
- Countries with high levels of social support and GDP per capita tend to have higher happiness scores.
- Generosity and life expectancy have a weaker correlation with happiness compared to economic stability and social support.

---

## 3. Media Dataset (`media.csv`)

### Dataset Description
The `media.csv` dataset contains information about media consumption habits across different regions, including time spent on television, internet, and print media.

### Key Features:
- **Region**: The geographic region of the media consumption data.
- **Television Hours**: Average hours per week spent watching television.
- **Internet Hours**: Average hours per week spent using the internet.
- **Print Media Hours**: Average hours per week spent reading print media.
- **Total Hours**: Total average hours spent on all media types combined.

### Analysis and Insights:
- **Summary Statistics**: Television and internet consumption dominate, with the average total media consumption around 25-30 hours per week.
- **Missing Values**: A few data points are missing for print media consumption, which affects the analysis of overall media habits.
- **Correlation**: A negative correlation exists between television hours and internet usage, suggesting that people who spend more time on one tend to spend less on the other.
- **Outliers**: A few regions report extremely high television consumption (over 50 hours/week), which may be indicative of specific cultural or regional preferences.
- **Trend Analysis**: Trend analysis shows a slight decrease in television hours and a corresponding increase in internet consumption, which could be attributed to the rise in digital media.

### Key Findings:
- Regions with higher internet consumption tend to have lower television viewing hours.
- A significant portion of total media consumption is dominated by internet usage in younger populations.

---

## Conclusion
This repository provides a comprehensive analysis of three distinct datasets, highlighting key trends, correlations, outliers, and other valuable insights. The findings offer actionable recommendations for further research and improvements in various fields, from book publishing and reader preferences to national happiness policies and media consumption habits.

For more detailed results, refer to the individual dataset analysis and visualizations saved in the respective folders.

---
