import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

ds = pd.read_csv("F:/OpenCV/titanic/Titanic-Dataset.csv")
ds.head()
print(ds.describe())
num_col = ds.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = ds.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {num_col}")
print(f"Categorical columns: {cat_cols}")

#plots for the numerical features
ds[num_col].hist(figsize=(10, 6), bins=30)
plt.suptitle("Histograms of Numerical Features")
plt.show()
ds[num_col].plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8), title="Boxplots of Numerical Features")
plt.show()
fig, axes = plt.subplots(nrows=len(cat_cols)//2 + 1, ncols=2, figsize=(15, len(cat_cols) * 2))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sb.countplot(y=ds[col], order=ds[col].value_counts().index[:10], ax=axes[i])
    axes[i].set_title(f"Count plot of {col}")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sb.heatmap(ds[num_col].corr(), annot=True, cmap="BrBG", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sb.barplot(x=col, y='Survived', data=ds)
    plt.title(f"Survival Rate by {col}")
    plt.xticks()
    plt.show()
