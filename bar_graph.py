import matplotlib.pyplot as plt

# Data from the table
categories = ['Wolfram graphs (training)', 'Wolfram graphs (testing)', 'Random graphs', 'Independent set = 2 graphs']
accuracy = [0.88, 0.89, 0.24, 0.14]  # Values for the single row

# Create a bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, accuracy, color='skyblue', edgecolor='black')

# Annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # vertical offset
                 textcoords='offset points',
                 ha='center', va='bottom')

# Add title and labels
plt.title('Model Accuracy on Different Graph Sets (Trained on Random Graphs for 9950 Epochs)')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)  # Show full scale from 0 to 1.1

plt.xticks(rotation=15)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
