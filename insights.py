import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

ds = pd.read_csv("F:/OpenCV/titanic/Titanic-Dataset.csv")

#feature engineering implementation using family size vs survival rate
ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

heatmap_data = ds.pivot_table(index="FamilySize", columns="Survived", aggfunc="size", fill_value=0)

plt.figure(figsize=(10, 8))
sb.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm")
plt.title("Heatmap of Family Size vs Survival")
plt.xlabel("Survived (1 = Yes, 0 = No)")
plt.ylabel("Family Size")
plt.show()
