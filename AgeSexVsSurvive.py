import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

ds = pd.read_csv("F:/OpenCV/titanic/Titanic-Dataset.csv")

#survival vs passenger class
sb.barplot(x="Pclass", y="Survived", data=ds, ci=None)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


sb.countplot(x="Pclass", hue="Survived", data=ds)
plt.title("Passenger Class vs. Survival Count")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(["Did Not Survive", "Survived"])
plt.show()

#sex vs survival
sb.barplot(x="Sex", y="Survived", data=ds, ci=None)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

sb.boxplot(x="Survived", y="Age", data=ds)
plt.title("Boxplot of Age vs. Survival")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

