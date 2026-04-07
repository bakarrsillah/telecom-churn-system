import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000  # number of customers

data = []

for i in range(n):
    customer_id = i + 1
    age = np.random.randint(18, 70)
    location = np.random.choice(['Freetown', 'Bo', 'Kenema', 'Makeni'])
    tenure = np.random.randint(1, 60)

    calls_per_day = np.random.randint(0, 20)
    avg_call_duration = np.random.uniform(1, 10)

    data_usage = np.random.uniform(50, 2000)

    monthly_charge = np.random.uniform(5, 100)
    payment_delay = np.random.randint(0, 30)

    complaints = np.random.randint(0, 5)
    last_activity = np.random.randint(0, 60)

    # 🎯 CHURN LOGIC (IMPORTANT)
    churn = 0

    if complaints > 2 or last_activity > 30:
        churn = 1
    elif calls_per_day < 3 and data_usage < 200:
        churn = 1
    elif tenure > 24 and complaints == 0:
        churn = 0

    data.append([
        customer_id, age, location, tenure,
        avg_call_duration, calls_per_day,
        data_usage, monthly_charge,
        payment_delay, complaints,
        last_activity, churn
    ])

df = pd.DataFrame(data, columns=[
    "customer_id","age","location","tenure_months",
    "avg_call_duration","calls_per_day",
    "data_usage_mb_per_day","monthly_charge",
    "payment_delay_days","num_complaints",
    "days_since_last_activity","churn"
])

df.to_csv("data/raw.csv", index=False)

print("✅ Dataset generated: data/raw.csv")