# Country Data Analysis Project

## Overview
This project performs comprehensive analysis of country-level data including population, GDP (market PPP), surface area, and derived GDP per capita metrics. The analysis creates ranked tables, calculates combined scores, and provides insights into country performance across multiple dimensions.

## Data Source
The primary data source for this analysis is the [World Bank DataBank - World Development Indicators](https://databank.worldbank.org/source/world-development-indicators#), which provides comprehensive global development data from the World Bank's extensive database of development indicators.

## What Was Accomplished

### 1. Data Loading and Processing
- **CSV Import**: Reads `country-population-gdp-area-2024-2020.csv` containing World Bank development indicators
- **Data Cleaning**: Handles missing values (".." converted to NaN) and converts to numeric format
- **Year Coverage**: Analyzes data from 2020-2024 with calculated averages

### 2. Table Creation and Analysis
Created four separate analytical tables:

#### Table 1: Population Data (SP.POP.TOTL)
- Population totals for all countries
- 2020-2024 yearly data with calculated averages
- Ranked by population size

#### Table 2: GDP (Market PPP) Data (NY.GDP.MKTP.PP.CD)
- GDP in current international dollars (PPP)
- Purchasing Power Parity adjusted for cost of living differences
- Ranked by economic size

#### Table 3: Surface Area Data (AG.SRF.TOTL.K2)
- Land area in square kilometers
- Geographic size comparison across countries
- Ranked by land area

#### Table 4: GDP Per Capita (Derived)
- Calculated as: GDP (Market PPP) / Population
- Economic efficiency per person
- Ranked by individual prosperity

### 3. Ranking and Scoring System
- **Individual Rankings**: Each metric ranked from highest to lowest
- **Combined Scoring**: Four-metric scoring system considering:
  - Population rank
  - GDP (Market PPP) rank
  - Surface area rank
  - GDP per capita rank
- **Score Calculation**: Lower ranks = higher scores, summed across all metrics

### 4. Advanced Data Integration
- **Subfolder CSV Discovery**: Automatically finds and reads CSV files ending with `_Data.csv` from all subdirectories
- **Region Classification**: Adds REGION column based on which data source contains each country
- **Folder-based Naming**: Uses folder names as DataFrame identifiers for easy reference

### 5. Output and Export
- **Multiple CSV Formats**:
  - `country_combined_scores.csv` - Basic rankings and scores
  - `country_combined_scores_with_region.csv` - Includes region classification
  - `final_country_analysis_table.csv` - Comprehensive final analysis
- **Data Visualization**: Displays top/bottom performers, score distributions, and sample data

## Data Structure

### Input Data
- **Source**: [World Bank DataBank - World Development Indicators](https://databank.worldbank.org/source/world-development-indicators#)
- **Format**: CSV with columns for Country Name, Country Code, Series Name, Series Code, and yearly values
- **Metrics**: Population, GDP (Market PPP), Surface Area
- **Time Period**: 2020-2024

### Output Tables
Each table includes:
- Country identification (Name, Code)
- Series information (Name, Code)
- Yearly values (2020-2024)
- Calculated averages
- Rankings and scores

## Key Features

### Functional Programming Approach
- Pure functions with clear input/output
- No hidden state changes
- Minimal, focused code changes
- Strong typing and error handling

### Comprehensive Analysis
- Multi-dimensional country comparison
- Statistical summaries and distributions
- Regional classification and grouping
- Export-ready data formats

### Error Handling
- Graceful handling of missing data
- Clear error messages and logging
- Robust CSV reading and processing

## Usage

### Running the Analysis
1. **Cell 1**: Load and process main CSV data
2. **Cell 2**: Create separate tables and add average columns
3. **Cell 3**: Rank all tables by average values
4. **Cell 4**: Calculate combined scores across all metrics
5. **Cell 5**: Discover and load additional CSV files from subfolders
6. **Cell 6**: Add region classification based on data sources
7. **Cell 7**: Export final comprehensive table

### Output Files
- `country_combined_scores.csv` - Basic analysis
- `country_combined_scores_with_region.csv` - With region data
- `final_country_analysis_table.csv` - Complete analysis

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **pathlib**: File path handling
- **IPython.display**: Enhanced data display
- **os**: Operating system interface

### Data Processing
- Missing value handling (".." → NaN)
- Numeric conversion with error handling
- Average calculations across time periods
- Ranking and scoring algorithms

### File Management
- Recursive subfolder search
- Automatic CSV detection and loading
- Multiple export formats
- Memory usage reporting

## Results and Insights

This analysis provides:
- **Country Rankings**: Performance across multiple dimensions
- **Combined Scores**: Overall country performance metrics
- **Regional Classification**: Data source-based grouping
- **Statistical Summaries**: Distributions and extremes
- **Export-Ready Data**: Clean, structured CSV outputs

The project successfully transforms raw World Bank data into actionable insights, enabling comprehensive country comparison and analysis across population, economic, geographic, and efficiency dimensions.
