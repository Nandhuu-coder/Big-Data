# Import required libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: Read CSV file
# -------------------------------

# Load dataset
data = pd.read_csv("data/transactions.csv")

# Display dataset
print("Dataset:\n")
print(data)

# -------------------------------
# STEP 2: Convert items into list
# -------------------------------

# Convert each transaction into list
transactions = []

for item in data['Items']:
    transactions.append(item.split(','))

print("\nTransactions:\n")
print(transactions)

# -------------------------------
# STEP 3: Apply Transaction Encoder
# -------------------------------

# Convert transaction data into binary format
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

# Create DataFrame
df = pd.DataFrame(te_data, columns=te.columns_)

print("\nEncoded Data:\n")
print(df)

# -------------------------------
# STEP 4: Apply Apriori Algorithm
# -------------------------------

# Find frequent itemsets
frequent_items = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:\n")
print(frequent_items)

# -------------------------------
# STEP 5: Generate Association Rules
# -------------------------------

# Create rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)

print("\nAssociation Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

# -------------------------------
# STEP 6: Visualization
# -------------------------------

# Count frequency of items
item_counts = df.sum().sort_values(ascending=False)

# Plot bar graph
plt.figure(figsize=(8,5))
item_counts.plot(kind='bar')

# Graph labels
plt.title("Item Frequency")
plt.xlabel("Items")
plt.ylabel("Count")

# Save graph image
plt.savefig("images/output_chart.png")

# Show graph
plt.show()

print("\nChart saved in images folder!")