import random
import pandas as pd

def generate_course_engagement_data(num_samples=1000):
    data = []
    for i in range(1, num_samples + 1):
        student = {
            "student_id": i,
            "logins_per_week": random.randint(1, 10),
            "videos_watched": random.randint(0, 30),
            "time_spent_on_platform": random.randint(1, 20),
            "avg_quiz_score": random.randint(50, 100)
        }
        data.append(student)
    
    return pd.DataFrame(data)

# Generate the synthetic dataset
course_engagement_data = generate_course_engagement_data()

# Display the first few rows
print(course_engagement_data.head())

# Save to CSV
course_engagement_data.to_csv("synthetic_course_engagement_data.csv", index=False)
print("\nDataset saved to 'synthetic_course_engagement_data.csv'")

# Basic statistics
print("\nBasic Statistics:")
print(course_engagement_data.describe())
