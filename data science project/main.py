import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

class DataProcessor:
    def __init__(self):
        self.student_profile = None
        self.course_engagement = None
        self.historical_data = None
        self.combined_data = None
        self.X = None
        self.y = None
        self.le_gender = LabelEncoder()
        self.le_major = LabelEncoder()
        self.le_region = LabelEncoder()
        self.gender_categories = ["Male", "Female"]
        self.region_categories = ["West Bengal", "Delhi", "Karnataka", "Maharashtra", "Tamil Nadu", 
                                  "Uttar Pradesh", "Gujarat", "Rajasthan", "Madhya Pradesh", "Kerala"]

    def load_data(self):
        self.student_profile = pd.read_csv("synthetic_student_data.csv")
        self.course_engagement = pd.read_csv("synthetic_course_engagement_data.csv")
        self.historical_data = pd.read_csv("synthetic_historical_data.csv")

    def combine_data(self):
        self.combined_data = self.student_profile.merge(self.course_engagement, on="student_id", how="inner")
        self.combined_data = self.combined_data.merge(self.historical_data, on="student_id", how="inner")

    def preprocess_data(self):
        self.combined_data['gender'] = self.le_gender.fit_transform(self.combined_data['gender'])
        self.combined_data['major'] = self.le_major.fit_transform(self.combined_data['major'])
        self.combined_data['region'] = self.le_region.fit_transform(self.combined_data['region'])

        self.combined_data['completed'] = (self.combined_data['courses_completed'] / self.combined_data['courses_started'] > 0.8).astype(int)

        features = ['age', 'gender', 'year', 'region', 'logins_per_week', 'videos_watched', 
                    'time_spent_on_platform', 'avg_quiz_score', 'courses_completed', 'courses_started']
        self.X = self.combined_data[features]
        self.y = self.combined_data['completed']

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_feature_importance(self):
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return feature_importance

class Visualizer:
    @staticmethod
    def plot_feature_importance(feature_importance):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    @staticmethod
    def plot_evaluation_metrics(metrics):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title('Evaluation Metrics')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

class StudentSuccessPredictor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = None
        self.visualizer = Visualizer()

    def run_analysis(self):
        self.data_processor.load_data()
        self.data_processor.combine_data()
        self.data_processor.preprocess_data()

        print("\nAvailable Gender Categories:", self.data_processor.gender_categories)
        print("Available Region Categories:", self.data_processor.region_categories)

        self.model_trainer = ModelTrainer(self.data_processor.X, self.data_processor.y)
        self.model_trainer.train_model()
        metrics = self.model_trainer.evaluate_model()

        feature_importance = self.model_trainer.get_feature_importance()
        print("\nFeature Importance:")
        print(feature_importance)

        self.visualizer.plot_feature_importance(feature_importance)
        self.visualizer.plot_confusion_matrix(self.model_trainer.y_test, self.model_trainer.model.predict(self.model_trainer.X_test))
        self.visualizer.plot_evaluation_metrics(metrics)

    def predict_student_success(self, student_data):
        original_gender = student_data['gender'].iloc[0]
        original_region = student_data['region'].iloc[0]
        
        if original_gender not in self.data_processor.gender_categories:
            print(f"Warning: Unknown gender category '{original_gender}'. Using a placeholder value.")
            print(f"Available gender categories: {self.data_processor.gender_categories}")
            student_data['gender'] = -1
        else:
            student_data['gender'] = self.data_processor.le_gender.transform([original_gender])[0]

        if original_region not in self.data_processor.region_categories:
            print(f"Warning: Unknown region category '{original_region}'. Using a placeholder value.")
            print(f"Available region categories: {self.data_processor.region_categories}")
            student_data['region'] = -1
        else:
            student_data['region'] = self.data_processor.le_region.transform([original_region])[0]

        required_features = self.data_processor.X.columns
        for feature in required_features:
            if feature not in student_data:
                raise ValueError(f"Missing required feature: {feature}")

        student_data = student_data[required_features]

        return self.model_trainer.model.predict(student_data)

def get_user_input(gender_categories, region_categories):
    print("\nEnter student data:")
    student_data = {}
    student_data['age'] = int(input("Age: "))
    print(f"Available gender categories: {gender_categories}")
    student_data['gender'] = input("Gender: ")
    student_data['year'] = int(input("Year of study: "))
    print(f"Available region categories: {region_categories}")
    student_data['region'] = input("Region: ")
    student_data['logins_per_week'] = int(input("Logins per week: "))
    student_data['videos_watched'] = int(input("Videos watched: "))
    student_data['time_spent_on_platform'] = float(input("Time spent on platform (hours): "))
    student_data['avg_quiz_score'] = float(input("Average quiz score: "))
    student_data['courses_completed'] = int(input("Courses completed: "))
    student_data['courses_started'] = int(input("Courses started: "))
    return pd.DataFrame([student_data])

if __name__ == "__main__":
    predictor = StudentSuccessPredictor()
    predictor.run_analysis()

    while True:
        user_input = get_user_input(predictor.data_processor.gender_categories, predictor.data_processor.region_categories)
        prediction = predictor.predict_student_success(user_input)
        print(f"\nPredicted outcome for student: {'Likely to complete' if prediction[0] == 1 else 'At risk of dropping out'}")
        
        continue_prediction = input("\nDo you want to predict for another student? (yes/no): ")
        if continue_prediction.lower() != 'yes':
            break

print("Thank you for using the Student Success Predictor!")