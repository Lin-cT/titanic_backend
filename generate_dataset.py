import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

# Adjust age mean and std
age_mean = 30                   # Mean age
age_std = 10                    # Increased variability in age
age = np.random.normal(age_mean, age_std, num_samples)

# Adjust stress level mean and std
stress_level_mean = 5           # Mean stress level
stress_level_std = 2            # Increased variability in stress level
stress_level = np.random.normal(stress_level_mean, stress_level_std, num_samples)

# Adjust exercise hours mean and std
exercise_hours_mean = 1.5       # Mean daily exercise hours
exercise_hours_std = 0.5        # Increased variability in daily exercise hours
exercise_hours = np.random.normal(exercise_hours_mean, exercise_hours_std, num_samples)

# Adjust sleep hours mean and std
sleep_hours_mean = 8            # Mean daily sleep hours
sleep_hours_std = 1             # Increased variability in daily sleep hours
sleep_hours = np.random.normal(sleep_hours_mean, sleep_hours_std, num_samples)

# Calculate probability of developing depression based on the factors
stress_factor = np.maximum(0, (stress_level - 6) * 2)
sleep_factor = np.maximum(0, (8 - sleep_hours) * 3)
exercise_factor = np.maximum(0, (1 - exercise_hours) * 1.5)
age_factor = np.maximum(0, age - 35)

probability = (stress_factor + sleep_factor + exercise_factor + age_factor) / 30

# Create DataFrame without 'Depression' column
data = pd.DataFrame({
    'Age': age,
    'Stress Level': stress_level,
    'Daily Exercise Hours': exercise_hours,
    'Daily Sleep Hours': sleep_hours,
    'Probability of Developing Depression': probability
})

# Save DataFrame to CSV
data.to_csv('depression_dataset.csv', index=False)


