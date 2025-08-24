#%% 
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from IPython.display import display


def read_country_data(csv_path: str = "country-population-gdp-area-2024-2020.csv") -> pd.DataFrame:
    """
    Read the country population, GDP, and area data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the country data
    """
    # Get the absolute path to the CSV file in the current directory
    current_dir = Path.cwd()
    csv_file_path = current_dir / csv_path
    
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print(f"Successfully read CSV file: {csv_file_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    display(df.head())
    
    return df

# Read the country data
try:
    country_data = read_country_data()
    print(f"\nTotal countries: {len(country_data['Country Name'].unique())}")
    print(f"Data years available: {[col for col in country_data.columns if '[' in col]}")
    
except Exception as e:
    print(f"Error reading CSV file: {e}")

# %%
# Create three separate tables based on series codes
print("="*60)
print("CREATING THREE SEPARATE TABLES BY SERIES CODE")
print("="*60)

# Helper function to add average column
def add_average_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an average column to the dataframe"""
    # Replace ".." with NaN
    year_columns = ['2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]', '2024 [YR2024]']
    
    # Convert to numeric, replacing ".." with NaN
    for col in year_columns:
        df[col] = pd.to_numeric(df[col].replace('..', pd.NA), errors='coerce')
    
    # Calculate average across years for each country (row-wise)
    df['Average_2020_2024'] = df[year_columns].mean(axis=1)
    
    return df

# Table 1: Population data (SP.POP.TOTL)
population_table = country_data[country_data['Series Code'] == 'SP.POP.TOTL'].copy()
population_table_with_avg = add_average_column(population_table)
print(f"\n1. POPULATION TABLE (SP.POP.TOTL) WITH AVERAGE COLUMN:")
print(f"   Shape: {population_table_with_avg.shape}")
print(f"   Columns: {list(population_table_with_avg.columns)}")
display(population_table_with_avg.head())

# Table 2: GDP (market PPP) data (NY.GDP.MKTP.PP.CD)
gdp_table = country_data[country_data['Series Code'] == 'NY.GDP.MKTP.PP.CD'].copy()
gdp_table_with_avg = add_average_column(gdp_table)
print(f"\n2. GDP (MARKET PPP) TABLE (NY.GDP.MKTP.PP.CD) WITH AVERAGE COLUMN:")
print(f"   Shape: {gdp_table_with_avg.shape}")
print(f"   Columns: {list(gdp_table_with_avg.columns)}")
display(gdp_table_with_avg.head())

# Table 3: Surface area data (AG.SRF.TOTL.K2)
area_table = country_data[country_data['Series Code'] == 'AG.SRF.TOTL.K2'].copy()
area_table_with_avg = add_average_column(area_table)
print(f"\n3. SURFACE AREA TABLE (AG.SRF.TOTL.K2) WITH AVERAGE COLUMN:")
print(f"   Shape: {area_table_with_avg.shape}")
print(f"   Columns: {list(area_table_with_avg.columns)}")
display(area_table_with_avg.head())

# Table 4: GDP per capita (derived from GDP market PPP / Population)
print(f"\n4. GDP PER CAPITA TABLE (DERIVED FROM GDP MARKET PPP / POPULATION):")

# Create a function to calculate GDP per capita
def calculate_gdp_per_capita(gdp_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate GDP per capita by dividing average GDP by average population for each country"""
    
    # Get unique countries that exist in both tables
    common_countries = set(gdp_df['Country Name']) & set(population_df['Country Name'])
    
    gdp_per_capita_data = []
    
    for country in common_countries:
        # Get GDP data for this country
        gdp_country = gdp_df[gdp_df['Country Name'] == country].iloc[0]
        # Get population data for this country
        pop_country = population_df[population_df['Country Name'] == country].iloc[0]
        
        # Calculate GDP per capita using averages
        try:
            gdp_avg = pd.to_numeric(gdp_country['Average_2020_2024'], errors='coerce')
            pop_avg = pd.to_numeric(pop_country['Average_2020_2024'], errors='coerce')
            
            if pd.notna(gdp_avg) and pd.notna(pop_avg) and pop_avg > 0:
                gdp_per_capita_avg = gdp_avg / pop_avg
            else:
                gdp_per_capita_avg = pd.NA
        except:
            gdp_per_capita_avg = pd.NA
        
        gdp_per_capita_row = {
            'Country Name': country,
            'Country Code': gdp_country['Country Code'],
            'Series Name': 'GDP per capita (derived from GDP market PPP / Population)',
            'Series Code': 'DERIVED_GDP_PCAP',
            'Average_2020_2024': gdp_per_capita_avg
        }
        
        gdp_per_capita_data.append(gdp_per_capita_row)
    
    # Create dataframe
    gdp_per_capita_df = pd.DataFrame(gdp_per_capita_data)
    
    return gdp_per_capita_df

# Calculate GDP per capita table
gdp_per_capita_table = calculate_gdp_per_capita(gdp_table_with_avg, population_table_with_avg)

print(f"   Shape: {gdp_per_capita_table.shape}")
print(f"   Columns: {list(gdp_per_capita_table.columns)}")
print(f"   Sample GDP per capita values:")
display(gdp_per_capita_table[['Country Name', 'Average_2020_2024']].head(10))

# Rank GDP per capita table by average (descending - highest GDP per capita first)
gdp_per_capita_ranked = gdp_per_capita_table.sort_values('Average_2020_2024', ascending=False).reset_index(drop=True)
print(f"\n   GDP PER CAPITA RANKING (Highest to Lowest):")
print(f"   Top 10 countries by average GDP per capita:")
display(gdp_per_capita_ranked[['Country Name', 'Average_2020_2024']].head(10))
print(f"   Bottom 10 countries by average GDP per capita:")
display(gdp_per_capita_ranked[['Country Name', 'Average_2020_2024']].tail(10))

# Summary of all tables
print(f"\n" + "="*60)
print("SUMMARY:")
print(f"Population table: {len(population_table_with_avg)} countries (including average)")
print(f"GDP table: {len(gdp_table_with_avg)} countries (including average)")
print(f"Area table: {len(area_table_with_avg)} countries (including average)")
print(f"GDP per capita table: {len(gdp_per_capita_table)} countries (including average)")
print("="*60)

# %%
# Rank all tables by average values
print("="*60)
print("RANKING TABLES BY AVERAGE VALUES")
print("="*60)

# Rank Population table by average (descending - highest population first)
population_ranked = population_table_with_avg.sort_values('Average_2020_2024', ascending=False).reset_index(drop=True)
print(f"\n1. POPULATION RANKING (Highest to Lowest):")
print(f"   Top 10 countries by average population:")
display(population_ranked[['Country Name', 'Average_2020_2024']].head(10))
print(f"   Bottom 10 countries by average population:")
display(population_ranked[['Country Name', 'Average_2020_2024']].tail(10))

# Rank GDP table by average (descending - highest GDP first)
gdp_ranked = gdp_table_with_avg.sort_values('Average_2020_2024', ascending=False).reset_index(drop=True)
print(f"\n2. GDP (MARKET PPP) RANKING (Highest to Lowest):")
print(f"   Top 10 countries by average GDP (market PPP):")
display(gdp_ranked[['Country Name', 'Average_2020_2024']].head(10))
print(f"   Bottom 10 countries by average GDP (market PPP):")
display(gdp_ranked[['Country Name', 'Average_2020_2024']].tail(10))

# Rank Area table by average (descending - largest area first)
area_ranked = area_table_with_avg.sort_values('Average_2020_2024', ascending=False).reset_index(drop=True)
print(f"\n3. SURFACE AREA RANKING (Largest to Smallest):")
print(f"   Top 10 countries by average surface area:")
display(area_ranked[['Country Name', 'Average_2020_2024']].head(10))
print(f"   Bottom 10 countries by average surface area:")
display(area_ranked[['Country Name', 'Average_2020_2024']].tail(10))

# Overall statistics
print(f"\n" + "="*60)
print("OVERALL STATISTICS:")
print(f"Population - Highest: {population_ranked.iloc[0]['Country Name']} ({population_ranked.iloc[0]['Average_2020_2024']:,.0f})")
print(f"Population - Lowest: {population_ranked.iloc[-1]['Country Name']} ({population_ranked.iloc[-1]['Average_2020_2024']:,.0f})")
print(f"GDP - Highest: {gdp_ranked.iloc[0]['Country Name']} (${gdp_ranked.iloc[0]['Average_2020_2024']:,.2f})")
print(f"GDP - Lowest: {gdp_ranked.iloc[-1]['Country Name']} (${gdp_ranked.iloc[-1]['Average_2020_2024']:,.2f})")
print(f"GDP Per Capita - Highest: {gdp_per_capita_ranked.iloc[0]['Country Name']} (${gdp_per_capita_ranked.iloc[0]['Average_2020_2024']:,.2f})")
print(f"GDP Per Capita - Lowest: {gdp_per_capita_ranked.iloc[-1]['Country Name']} (${gdp_per_capita_ranked.iloc[-1]['Average_2020_2024']:,.2f})")
print(f"Area - Largest: {area_ranked.iloc[0]['Country Name']} ({area_ranked.iloc[0]['Average_2020_2024']:,.0f} sq km)")
print(f"Area - Smallest: {area_ranked.iloc[-1]['Country Name']} ({area_ranked.iloc[-1]['Average_2020_2024']:,.0f} sq km)")
print("="*60)

# %%
# Calculate combined score for each country based on ranks across all three tables
print("="*60)
print("COMBINED SCORE CALCULATION")
print("="*60)

# Create a function to get rank for each country
def get_country_rank(ranked_df: pd.DataFrame, country_name: str) -> int:
    """Get the rank of a country in a ranked dataframe"""
    try:
        return ranked_df[ranked_df['Country Name'] == country_name].index[0] + 1
    except:
        return len(ranked_df) + 1  # If country not found, assign worst rank

# Get all unique countries
all_countries = set(population_table_with_avg['Country Name']) | set(gdp_table_with_avg['Country Name']) | set(area_table_with_avg['Country Name']) | set(gdp_per_capita_table['Country Name'])

# Calculate combined scores
country_scores = []
for country in all_countries:
    pop_rank = get_country_rank(population_ranked, country)
    gdp_rank = get_country_rank(gdp_ranked, country)
    area_rank = get_country_rank(area_ranked, country)
    gdp_per_capita_rank = get_country_rank(gdp_per_capita_ranked, country)
    
    # Calculate combined score (lower rank = better, so we invert for scoring)
    # Population: higher is better, GDP: higher is better, Area: higher is better, GDP per capita: higher is better
    pop_score = len(population_ranked) - pop_rank + 1
    gdp_score = len(gdp_ranked) - gdp_rank + 1
    area_score = len(area_ranked) - area_rank + 1
    gdp_per_capita_score = len(gdp_per_capita_ranked) - gdp_per_capita_rank + 1
    
    # Combined score (sum of all four scores)
    combined_score = pop_score + gdp_score + area_score + gdp_per_capita_score
    
    country_scores.append({
        'Country Name': country,
        'Population_Rank': pop_rank,
        'GDP_Market_PPP_Rank': gdp_rank,
        'Area_Rank': area_rank,
        'GDP_Per_Capita_Rank': gdp_per_capita_rank,
        'Combined_Score': combined_score
    })

# Create score dataframe and sort by combined score
score_df = pd.DataFrame(country_scores)
score_df_sorted = score_df.sort_values('Combined_Score', ascending=False).reset_index(drop=True)

print(f"\nCombined Score Ranking (Higher score = better overall performance):")
print(f"Total countries analyzed: {len(score_df_sorted)}")
display(score_df_sorted.head(20))

print(f"\nTop 10 Countries by Combined Score:")
for i, row in score_df_sorted.head(10).iterrows():
    print(f"{i+1:2d}. {row['Country Name']:<25} Score: {row['Combined_Score']:3d} (Pop: {row['Population_Rank']:3d}, GDP PPP: {row['GDP_Market_PPP_Rank']:3d}, Area: {row['Area_Rank']:3d}, GDP PC: {row['GDP_Per_Capita_Rank']:3d})")

print(f"\nBottom 10 Countries by Combined Score:")
for i, row in score_df_sorted.tail(10).iterrows():
    print(f"{len(score_df_sorted)-9+i:2d}. {row['Country Name']:<25} Score: {row['Combined_Score']:3d} (Pop: {row['Population_Rank']:3d}, GDP PPP: {row['GDP_Market_PPP_Rank']:3d}, Area: {row['Area_Rank']:3d}, GDP PC: {row['GDP_Per_Capita_Rank']:3d})")

# Show score distribution
print(f"\n" + "="*60)
print("SCORE DISTRIBUTION:")
print(f"Highest Combined Score: {score_df_sorted.iloc[0]['Combined_Score']} ({score_df_sorted.iloc[0]['Country Name']})")
print(f"Lowest Combined Score: {score_df_sorted.iloc[-1]['Combined_Score']} ({score_df_sorted.iloc[-1]['Country Name']})")
print(f"Average Combined Score: {score_df_sorted['Combined_Score'].mean():.1f}")
print(f"Median Combined Score: {score_df_sorted['Combined_Score'].median():.1f}")
print("="*60)

# Export the final combined score table to CSV
print(f"\n" + "="*60)
print("EXPORTING FINAL TABLE TO CSV")
print("="*60)

# Export the sorted score dataframe to CSV
output_filename = "country_combined_scores.csv"
score_df_sorted.to_csv(output_filename, index=False)

print(f"\nFinal combined score table exported to: {output_filename}")
print(f"Table contains {len(score_df_sorted)} countries")
print(f"Columns: {list(score_df_sorted.columns)}")
print(f"File size: {score_df_sorted.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Display first few rows of the exported table
print(f"\nFirst 10 rows of the exported table:")
display(score_df_sorted.head(10))

print(f"\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)

# %%
# Read all CSV files ending with "_Data.csv" from all subfolders
print("="*60)
print("READING ALL CSV FILES ENDING WITH '_Data.csv'")
print("="*60)

import os
from pathlib import Path

def find_and_read_data_csvs(root_dir: str = ".") -> dict:
    """
    Find and read all CSV files ending with '_Data.csv' from all subfolders
    
    Args:
        root_dir: Root directory to search from
        
    Returns:
        Dictionary with folder names as keys and dataframes as values
    """
    csv_files = {}
    root_path = Path(root_dir)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('_Data.csv'):
                file_path = Path(root) / file
                # Get the folder name as the key
                folder_name = Path(root).name
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    csv_files[folder_name] = df
                    print(f"✓ Successfully read: {file_path}")
                    print(f"  Folder name (DataFrame key): {folder_name}")
                    print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"✗ Error reading {file_path}: {e}")
    
    return csv_files

# Find and read all data CSV files
print(f"\nSearching for CSV files ending with '_Data.csv' in: {os.getcwd()}")
data_csvs = find_and_read_data_csvs()

print(f"\n" + "="*60)
print("SUMMARY OF FOUND CSV FILES:")
print("="*60)

if data_csvs:
    print(f"Total CSV files found: {len(data_csvs)}")
    print(f"\nFile details:")
    
    for folder_name, df in data_csvs.items():
        print(f"\n📁 {folder_name}")
        print(f"   DataFrame key: {folder_name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample data:")
        display(df.head(3))
else:
    print("No CSV files ending with '_Data.csv' found in any subfolders")

print("="*60)

# %%
# Add REGION column to the final table based on which DataFrame each country comes from
print("="*60)
print("ADDING REGION COLUMN TO FINAL TABLE")
print("="*60)

def add_region_column(score_df: pd.DataFrame, data_csvs: dict) -> pd.DataFrame:
    """
    Add REGION column to the score dataframe based on which DataFrame each country appears in
    
    Args:
        score_df: The score dataframe with country names
        data_csvs: Dictionary of dataframes with folder names as keys
        
    Returns:
        Updated dataframe with REGION column
    """
    # Create a copy to avoid modifying the original
    updated_df = score_df.copy()
    updated_df['REGION'] = 'Unknown'  # Default value
    
    # Check each country against each dataframe
    for country in updated_df['Country Name']:
        for folder_name, df in data_csvs.items():
            if country in df['Country Name'].values:
                updated_df.loc[updated_df['Country Name'] == country, 'REGION'] = folder_name
                break  # Found the country, move to next country
    
    return updated_df

# Add REGION column to the final table
if 'data_csvs' in locals() and data_csvs:
    print("Adding REGION column based on found DataFrames...")
    
    # Update the score dataframe with REGION column
    score_df_with_region = add_region_column(score_df_sorted, data_csvs)
    
    print(f"\nUpdated final table shape: {score_df_with_region.shape}")
    print(f"Updated columns: {list(score_df_with_region.columns)}")
    
    # Show sample of updated table
    print(f"\nSample of updated final table with REGION column:")
    display(score_df_with_region.head(10))
    
    # Show region distribution
    print(f"\nRegion distribution:")
    region_counts = score_df_with_region['REGION'].value_counts()
    for region, count in region_counts.items():
        print(f"  {region}: {count} countries")
    
    # Export updated table to CSV
    output_filename_updated = "country_combined_scores_with_region.csv"
    score_df_with_region.to_csv(output_filename_updated, index=False)
    
    print(f"\n" + "="*60)
    print("UPDATED TABLE EXPORTED:")
    print(f"Filename: {output_filename_updated}")
    print(f"Total countries: {len(score_df_with_region)}")
    print(f"Regions found: {len(region_counts)}")
    print("="*60)
    
else:
    print("No DataFrames found. Please run the previous cell to load CSV files first.")

# %%

# %%
# Save the final table to CSV
print("="*60)
print("SAVING FINAL TABLE TO CSV")
print("="*60)

# Check if we have the updated table with region
if 'score_df_with_region' in locals():
    final_table = score_df_with_region
    print("Using updated table with REGION column")
elif 'score_df_sorted' in locals():
    final_table = score_df_sorted
    print("Using original combined scores table")
else:
    print("No final table found. Please run the previous cells first.")
    final_table = None

if final_table is not None:
    # Save to CSV
    output_filename = "final_country_analysis_table.csv"
    final_table.to_csv(output_filename, index=False)
    
    print(f"\nFinal table saved successfully!")
    print(f"Filename: {output_filename}")
    print(f"File location: {os.getcwd()}")
    print(f"Table shape: {final_table.shape}")
    print(f"Columns: {list(final_table.columns)}")
    print(f"File size: {final_table.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Display first few rows
    print(f"\nFirst 10 rows of the final table:")
    display(final_table.head(10))
    
    # Display last few rows
    print(f"\nLast 10 rows of the final table:")
    display(final_table.tail(10))
    
    print(f"\n" + "="*60)
    print("CSV EXPORT COMPLETE!")
    print("="*60)
else:
    print("Cannot save table - no data available")

# %%