import random
import pandas as pd

def generate_historical_data(num_samples=1000):
    data = []
    for i in range(1, num_samples + 1):
        courses_started = random.randint(1, 10)
        student = {
            "student_id": i,
            "courses_completed": random.randint(0, courses_started),
            "courses_started": courses_started,
            "avg_score_across_courses": random.randint(50, 100)
        }
        data.append(student)
    
    return pd.DataFrame(data)

# Generate the synthetic dataset
historical_data = generate_historical_data()

# Display the first few rows
print(historical_data.head())

# Save to CSV
historical_data.to_csv("synthetic_historical_data.csv", index=False)
print("\nDataset saved to 'synthetic_historical_data.csv'")

# Basic statistics
print("\nBasic Statistics:")
print(historical_data.describe())