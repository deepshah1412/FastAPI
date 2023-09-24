#Importing the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#st.header("Salary Predictor of Data Scientist!")
# Libraries for Data visualization
from matplotlib import pyplot as plot
import statsmodels.api as sm
import seaborn as sns   #statistical graphics
#KDD process

#Reading the dataset
data_set=pd.read_csv("salaries.csv.csv")
data_set.head(5)
#Data Cleaning (Cleaning in case of Missing values)
data_set.isnull().sum()
#fill the missing values by mode method
data_set.fillna({"work_year":data_set['work_year'].mode()[0],"employment_type":data_set['employment_type'].mode()[0],"remote_ratio":data_set['remote_ratio'].mode()[0]},inplace=True)
data_set.shape
data_set.isnull().sum()
data_set.head()
#data transformation
LE = LabelEncoder()
CateList = data_set.select_dtypes(include="object").columns

for i in CateList:
    data_set[i] = LE.fit_transform(data_set[i])
data_set.head(5)
#Setting the value for X and Y
#data preprocessing
Y = data_set.salary_in_usd
#X = data_set[['experience_level','company_size','work_year','remote_ratio','employment_type','job_title','company_location']]
X = data_set[['remote_ratio','work_year','employment_type','experience_level','salary','salary_currency','job_title','company_location','employee_residence','company_size']]
#corelaated
def preprocessing(X):    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    variables = X_scaled
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
    vif["Features"] = X.columns
    

preprocessing(X)
X.drop(['employment_type','salary_currency','company_location','employee_residence','salary'], axis=1, inplace=True)
preprocessing(X) 
#data mining
# Splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30,random_state=100)
#Fitting the Multiple Linear Regression model
regression = LinearRegression()
regression.fit(x_train,y_train)
#Prediction of test set
y_predict = regression.predict(x_test)
plot.scatter(y_test,y_predict)       
plot.xlabel('y_test', fontsize=18)                        
plot.ylabel('y_predict', fontsize=16)

#Pattern Evaluation/Interpretation
sns.regplot(x=y_test,y=y_predict,ci=None,color='blue')
#Knowledge representation:
model_1 = sm.OLS(y_train, x_train).fit()
# print(model_1.summary())
# print(f"R squared is: {model_1.rsquared*100}")

new_x = []
dict={1:22,2:17,3:12,4:41,5:38,6:48,7:21,8:16,9:7,10:41,11:46,12:1,13:19,14:25,15:15,16:26,17:6}
a=int(input("Please select remote ratio\n .0\n 50\n 100\n"))
new_x.append(a)
b=int(input("work_year: "))
new_x.append(b)
c=int(input("Please select your experience level\n0.Entry level\n1.Executive-level\n2.Mid level\n3.Senior-level\n"))
new_x.append(c)
e=int(input("\n1.Data Scientist\n2.Data Engineer\n3.Data Analyst\n4.Machine Learning Engineer\n5.Research Scientist\n6.Data Science Manager\n7.Data Architect\n8.Big Data Engineer\n9.Machine Learning Scientist\n10.Principal Data Scientist\n11.AI Scientist\n12.Data Science Consultant\n13.Director of Data Science\n14.Data Analytics Manager\n15.ML Engineer\n16.Computer Vision Engineer\n17.BI Data Analyst\nPlease select a job title from the above: "))
e1=dict[e]
new_x.append(e1)
f=int(input("Please select the company size\n0.small\n1.medium\n2.big\n"))
new_x.append(f)

#new_x=[100.0,2022.0,3,140000,22,1]
df=pd.DataFrame([new_x],columns=['remote_ratio','work_year','experience_level','job_title','company_size'])
df
y_predict = regression.predict(df)
y_predict=int(y_predict)
print(f"Your salary for given criteria is: ${y_predict}")
plot.show()

#now converting the model into pickel file
# import pickle
# filename='salary_predict.pkl'
# pickle_out=open(filename,'wb')
# pickle.dump(regression,pickle_out) # regression was the name of model
# pickle_out.close()