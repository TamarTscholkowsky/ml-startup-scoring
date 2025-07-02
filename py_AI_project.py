
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# מודלים לרגרסיה
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
# מודלים לסיווג
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#טעינת הdata Frame
df = pd.read_csv(r"C:\Users\משתמש\Documents\phyton\py_AI\project.csv")
print(df.head())

#עיבוד הנתונים והכנתם לאימון
# המרת משתנים קטגוריאליים לעמודות בינאריות באמצעות `get_dummies()`
df = pd.get_dummies(df, columns=["Location", "Marketing Strategy"], drop_first=False)
#שיטה נוספת לקידוד אבל לא השתמשתי איתה כי הערכים בשיטת הקידוד תלוים אחד בשני
# label_encoder = LabelEncoder()
# df["Location"] = label_encoder.fit_transform(df["Location"])
# df["Marketing Strategy"] = label_encoder.fit_transform(df["Marketing Strategy"])
# df["Above $5M Profit?"] = label_encoder.fit_transform(df["Above $5M Profit?"])  # Yes = 1, No = 0

# נרמול הערכים המספריים בין 0 ל-1 באמצעות MinMaxScaler
#שיהיה יחסיות בין הערכים
scaler = MinMaxScaler()
# זיהוי עמודות מספריות בלבד (כדי למנוע נירמול על עמודות בינאריות שנוצרו מ־get_dummies)
numeric_columns = df.select_dtypes(include=["number"]).columns
# החלת נירמול על העמודות המספריות בלבד
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
# זיהוי העמודות המספריות (אחרי הקידוד)
numeric_columns = df.select_dtypes(include=["number"]).columns
# החלת הנירמול מחדש לאחר הקידוד
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

#############
#אימון המודל
# לעמודת הinput
X = df.drop(columns=["Total Revenue ($M)", "Company Name"])  # הסרת משתני הזיהוי והמשתנה לחיזוי
# עמודת output
y = df["Total Revenue ($M)"]
# פיצול הנתונים לסט אימון (80%) וסט בדיקה (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# נרמול נתונים-להפוך את המידע לסטרנדרטי
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# מילון שמכיל את המודלים
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

predictions = {}
model_results = {}  # משתנה נוסף לשמירת התחזיות של כל מודל

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # שמירת התחזיות
    predictions[name] = model.predict(X_test)
    model_results[name] = y_pred  # שמירת התחזיות
#########תצוגה
result_mse={}
result_r2={}
#תצוגת בכתב
def print_model_metrics(y_test,y_pred,model_name):
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    result_mse[model_name]=mse
    result_r2[model_name]=r2
    print(f'{model_name}:\nMSE:{mse:.2f}, R^2:{r2:.2f}\n')

for name, y_pred in predictions.items():
    print_model_metrics(y_test, y_pred, name)

#תצוגה ויזואלית
plt.figure(figsize=(15,10))
for i, (name, y_pred) in enumerate(predictions.items(), start=1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred)  # `y_pred` הוא מערך תחזיות לכל ערך ב-`y_test`
    plt.xlabel('Real Total Revenue ($M)')
    plt.ylabel('Predicted Total Revenue ($M)')
    plt.title(name)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.tight_layout()
plt.show()
#מסקנות סופיות
print("אם חשוב לך מספר שגיאות קטן")
name_modele="Linear Regression"
result=result_mse["Linear Regression"]
for name,val in result_mse.items():
    if val<result:
        name_modele =name
        result = result_mse[name]
print("תבחר במודל "+name_modele)
print("אם חשוב לך מספר הצלחות גבוה")
name_modele="Linear Regression"
result=result_r2["Linear Regression"]
for name,val in result_r2.items():
    if val<result:
        name_modele =name
        result = result_r2[name]
        print("תבחר את המודל"+name_modele)



