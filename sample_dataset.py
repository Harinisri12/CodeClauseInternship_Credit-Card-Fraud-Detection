import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

np.random.seed(42)

fake = Faker()

num_transactions = 10000

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dates = [fake.date_time_between(start_date, end_date) for _ in range(num_transactions)]

amounts = [round(random.uniform(1, 1000), 2) for _ in range(num_transactions)]

merchants = [fake.company() for _ in range(num_transactions)]

card_numbers = [fake.credit_card_number(card_type="mastercard") for _ in range(num_transactions)]

transaction_types = ['purchase', 'withdrawal', 'transfer']
types = [random.choice(transaction_types) for _ in range(num_transactions)]

fraud_labels = [1 if random.random() < 0.05 else 0 for _ in range(num_transactions)]

data = pd.DataFrame({
    'TransactionDate': dates,
    'Amount': amounts,
    'Merchant': merchants,
    'CardNumber': card_numbers,
    'TransactionType': types,
    'Class': fraud_labels
})

data.to_csv('credit_card_transactions.csv', index=False)

print(data.head())
