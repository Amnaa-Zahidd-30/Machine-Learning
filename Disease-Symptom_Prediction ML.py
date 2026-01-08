import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("training.csv")
print("Dataset Loaded")
print(df.head())
print("Dataset Shape:",df.shape)
print("Missing Values:\n",df.isnull().sum())
df.fillna(0,inplace=True)

diseasecount=df["prognosis"].value_counts()
topdisease=diseasecount[:10]
others=diseasecount[10:].sum()
topdisease["Others"]=others

plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
plt.pie(topdisease,labels=topdisease.index,autopct="%1.1f%%",startangle=140)
plt.title("Top 10 Diseases Distribution")
plt.axis("equal")
plt.subplot(1,2,2)
sns.barplot(x=topdisease.values,y=topdisease.index)
plt.title("Top 10 Most Frequent Diseases")
plt.xlabel("Cases")
plt.ylabel("Disease")
plt.tight_layout()
plt.show()

encoder=LabelEncoder()
df["prognosis"]=encoder.fit_transform(df["prognosis"])
X=df.drop("prognosis",axis=1)
y=df["prognosis"]

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

models={
"Logistic Regression":LogisticRegression(max_iter=1000),
"Random Forest":RandomForestClassifier(n_estimators=200,random_state=42),
"SVM":SVC(),
"K-Nearest Neighbors":KNeighborsClassifier(),
"SGD Classifier":SGDClassifier(max_iter=1000,tol=1e-3)
}

best_model=None
best_accuracy=0
for name,model in models.items():
    model.fit(Xtrain,ytrain)
    ypred=model.predict(Xtest)
    accuracy=accuracy_score(ytest,ypred)
    print(f"{name} Accuracy: {accuracy*100:.2f}%")
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        best_model=model
        best_model_name=name

symptoms=X.columns.tolist()
print("Available Symptoms:")
print(symptoms)

def predictdisease(usersymptoms):
    inputdata=np.zeros(len(symptoms))
    for s in usersymptoms:
        s=s.strip().lower()
        if s in symptoms:
            index=symptoms.index(s)
            inputdata[index]=1
    pred=best_model.predict([inputdata])[0]
    disease=encoder.inverse_transform([pred])[0]
    return disease

print("Note: This system is for educational purposes only and is NOT a medical diagnosis tool.")

userinput=input("\nEnter symptoms separated by comma: ").lower().split(",")
userinput=[s.strip() for s in userinput]
result=predictdisease(userinput)
print(f"Predicted Disease: {result} (Model Used: {best_model_name})")
