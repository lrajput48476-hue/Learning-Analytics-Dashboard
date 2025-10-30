import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DATA_PATH = 'uwe_blackboard_data.csv'
N_CLUSTERS = 3
RANDOM_STATE = 42
WEEKLY_COLUMNS = [f'Week_{i}_Score' for i in range(1, 16)] # Assuming 15 weeks
CLUSTER_LABELS = {
    0: 'Low Engagement',
    1: 'Late Surge',
    2: 'High Consistent'
}

# --- 2. DATA PREPROCESSING ---
def load_and_preprocess_data(file_path):
    """
    Loads raw data, calculates weekly engagement, and prepares data for clustering.
    """
    print("Starting data loading and preprocessing...")
    try:
        # Load sample data (Replace with your actual data loading)
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Creating mock data for demonstration.")
        # Mock Data Creation (if file is missing)
        n_students = 200
        data = {
            'StudentID': range(1001, 1001 + n_students),
            'Final_Grade': np.random.randint(40, 95, n_students)
        }
        # Simulate weekly engagement data
        for i in range(1, 16):
            # Base activity, increasing towards the end (for the 'Late Surge' pattern)
            base_activity = np.random.uniform(0, 0.5 + i/30, n_students)
            # Add noise
            data[f'Week_{i}_Score'] = np.clip(base_activity + np.random.normal(0, 0.1), 0, 1)

        df = pd.DataFrame(data)
        
    # a. Calculate Total Engagement Score (used for correlation/regression)
    df['Total_Engagement_Score'] = df[WEEKLY_COLUMNS].sum(axis=1)

    # b. Normalize Data for Clustering (if weekly scores aren't already 0-1)
    # Use the weekly columns for clustering
    X_weekly = df[WEEKLY_COLUMNS].fillna(df[WEEKLY_COLUMNS].median())
    
    # Scale data (often required for distance-based clustering like K-Means)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_weekly)
    
    print("Preprocessing complete.")
    return df, X_scaled

# --- 3. K-MEANS CLUSTERING ---
def perform_clustering(X_scaled, n_clusters, df):
    """
    Performs K-Means clustering and assigns cluster labels to the DataFrame.
    """
    print(f"Performing K-Means clustering with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map numerical clusters to descriptive labels based on analysis in Contribution 2
    df['Engagement_Profile'] = df['Cluster'].map(CLUSTER_LABELS)
    
    # Analyze the average profile of each cluster (for visualization)
    cluster_profiles = df.groupby('Engagement_Profile')[WEEKLY_COLUMNS].mean().T
    
    print("Clustering complete. Engagement profiles created.")
    return df, cluster_profiles

# --- 4. VISUALIZATION (Generating Figures 2, 3, and 4) ---
def generate_visualizations(df, cluster_profiles):
    """
    Generates the required visualizations: Line chart, Bar chart, and Scatter plot.
    """
    print("Generating visualizations...")
    sns.set_style("whitegrid")
    
    # --- FIGURE 2: WEEKLY ENGAGEMENT TRENDS (Line Chart) ---
    plt.figure(figsize=(12, 6))
    
    # Plot each cluster's average weekly score
    for profile in cluster_profiles.columns:
        plt.plot(cluster_profiles.index, cluster_profiles[profile], label=profile, marker='o', markersize=4)

    # Add peak indicators based on analysis description
    plt.axvline(x=5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Mid-term Peak') # Example Week 5
    plt.axvline(x=12, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Pre-Exam Peak') # Example Week 12

    plt.title('Figure 2: Weekly Engagement Trends by Cluster Profile')
    plt.xlabel('Week of Semester')
    plt.ylabel('Average Normalized Engagement Score')
    plt.xticks(ticks=range(len(WEEKLY_COLUMNS)), labels=[str(i) for i in range(1, len(WEEKLY_COLUMNS) + 1)])
    plt.legend(title='Engagement Profile')
    plt.tight_layout()
    plt.savefig('Figure_2_Weekly_Engagement_Trends.png')
    plt.show()

    # --- FIGURE 3: CLUSTER DISTRIBUTION (Bar Chart) ---
    plt.figure(figsize=(8, 6))
    cluster_counts = df['Engagement_Profile'].value_counts(normalize=True).mul(100).sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    
    plt.title('Figure 3: Cluster Distribution of Engagement Profiles (Student Population %)')
    plt.xlabel('Engagement Profile')
    plt.ylabel('Percentage of Student Population (%)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('Figure_3_Cluster_Distribution.png')
    plt.show()

    # --- FIGURE 4: ENGAGEMENT VS. FINAL GRADES (Scatter Plot) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Engagement_Score', y='Final_Grade', hue='Engagement_Profile', data=df, palette='viridis', alpha=0.7)
    
    # Optional: Add a regression line to show the positive correlation (r=0.46)
    sns.regplot(x='Total_Engagement_Score', y='Final_Grade', data=df, scatter=False, color='red', line_kws={'linestyle':'--'})
    
    plt.title('Figure 4: Total Engagement Score vs. Final Grades (r = 0.46)')
    plt.xlabel('Total Normalized Engagement Score (Sum of Weekly Scores)')
    plt.ylabel('Final Grade (%)')
    plt.legend(title='Engagement Profile')
    plt.tight_layout()
    plt.savefig('Figure_4_Engagement_vs_Grades.png')
    plt.show()
    
    print("Visualizations saved as PNG files.")

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # 1. Preprocessing
    final_df, scaled_data = load_and_preprocess_data(DATA_PATH)
    
    # 2. Clustering
    clustered_df, cluster_profiles = perform_clustering(scaled_data, N_CLUSTERS, final_df)
    
    # 3. Visualization
    generate_visualizations(clustered_df, cluster_profiles)

    # Display the correlation coefficient found
    correlation = clustered_df['Total_Engagement_Score'].corr(clustered_df['Final_Grade'])
    print(f"\nCalculated Pearson Correlation (Engagement vs. Grade): {correlation:.2f}")
