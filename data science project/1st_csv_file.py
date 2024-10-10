#creating the synthetic data
import random
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    majors = ["Computer Science", "Mechanical Engineering", "Environmental Science", 
              "Civil Engineering", "Electrical Engineering", "Biology", "Chemistry", 
              "Physics", "Mathematics", "Economics"]
    regions = ["West Bengal", "Delhi", "Karnataka", "Maharashtra", "Tamil Nadu", 
               "Uttar Pradesh", "Gujarat", "Rajasthan", "Madhya Pradesh", "Kerala"]
    
    data = []
    for i in range(1, num_samples + 1):
        student = {
            "student_id": i,
            "age": random.randint(18, 25),
            "gender": random.choice(["Male", "Female"]),
            "major": random.choice(majors),
            "year": random.randint(1, 4),
            "region": random.choice(regions)
        }
        data.append(student)
    
    return pd.DataFrame(data)

# Generate the synthetic dataset
synthetic_data = generate_synthetic_data()

# Display the first few rows
print(synthetic_data.head())

# Save to CSV
synthetic_data.to_csv("synthetic_student_data.csv", index=False)
print("\nDataset saved to 'synthetic_student_data.csv'")

# Basic statistics
print("\nBasic Statistics:")
print(synthetic_data.describe())

print("\nMajor Distribution:")
print(synthetic_data['major'].value_counts(normalize=True))

print("\nGender Distribution:")
print(synthetic_data['gender'].value_counts(normalize=True))