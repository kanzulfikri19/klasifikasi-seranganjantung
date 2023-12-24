import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

st.title("KLASIFIKASI DECISION TREE")

def load_data():
 data = pd.read_csv("aa.csv")
 return data
df=load_data()
if st.checkbox("Tampilkan data"):
    st.subheader("Heart Failure Dataset")
    st.write(df)
#Mendefinisikan fitur dan label
data = pd.read_csv("aa.csv")
feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction' , 'high_blood_pressure', 'platelets' ,'serum_creatinine','serum_sodium', 'sex','smoking','time']
X = data[feature_cols] # Features
y = data.DEATH_EVENT # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Mendeklarasikan algoritma decision tree
model = DecisionTreeClassifier (max_depth=4)
model.fit(X_train, y_train)

#mMenampilkan hasil confusion matrix
if st.checkbox("Tampilkan Plot Confusion Matrix"):
        st.header("Confussion Matrix")
        plot_confusion_matrix(model, X_test, y_test)  
        plt.show()
        st.pyplot(plt)
#Menampilkan hasil klasifikasi dan tree
if st.checkbox("Tampilkan hasil klasifikasi"):
    y_pred = model.predict (X_test)
    st.write(classification_report(y_test, y_pred))
if st.checkbox("Tampilkan tree"):
    dot_data  = tree.export_graphviz(model, out_file=None)
    st.graphviz_chart(dot_data)

#Melakukan prediksi


#ubah input data menjadi array
def user_input_features():
    age = st.sidebar.number_input('age', min_value=int(X.age.min()), max_value=int(X.age.max()),
                             value=int(X.age.mean()))
    anaemia = st.sidebar.number_input('anaemia', min_value=int(X.anaemia.min()), max_value=int(X.anaemia.max()),
                             value=int(X.anaemia.mean()))
    creatinine_phosphokinase = st.sidebar.number_input('creatinine_phosphokinase', min_value=int(X.creatinine_phosphokinase.min()), max_value=int(X.creatinine_phosphokinase.max()),
                             value=int(X.creatinine_phosphokinase.mean()))
    diabetes = st.sidebar.number_input('diabetes',min_value=int(X.diabetes.min()), max_value=int(X.diabetes.max()),
                             value=int(X.diabetes.mean()))
    ejection_fraction = st.sidebar.number_input('ejection_fraction', min_value=int(X.ejection_fraction.min()), max_value=int(X.ejection_fraction.max()),
                             value=int(X.ejection_fraction.mean()))
    high_blood_pressure = st.sidebar.number_input('high_blood_pressure', min_value=int(X.high_blood_pressure.min()), max_value=int(X.high_blood_pressure.max()),
                             value=int(X.high_blood_pressure.mean()))
    platelets = st.sidebar.number_input('platelets', min_value=int(X.platelets.min()), max_value=int(X.platelets.max()),
                             value=int(X.platelets.mean()))
    serum_creatinine = st.sidebar.number_input('serum_creatinine',min_value=int(X.serum_creatinine.min()), max_value=int(X.serum_creatinine.max()),
                             value=int(X.serum_creatinine.mean()))
    serum_sodium = st.sidebar.number_input('serum_sodium', min_value=int(X.serum_sodium.min()), max_value=int(X.serum_sodium.max()),
                             value=int(X.serum_sodium.mean()))
    sex = st.sidebar.number_input('sex', min_value=int(X.sex.min()), max_value=int(X.sex.max()),
                             value=int(X.sex.mean()))
    smoking = st.sidebar.number_input('smoking', min_value=int(X.smoking.min()), max_value=int(X.smoking.max()),
                             value=int(X.smoking.mean()))
    time = st.sidebar.number_input('time',min_value=int(X.time.min()), max_value=int(X.time.max()),
                             value=int(X.time.mean()))
                                                                                       
    data = {'age': age,
            'ejection_fraction': ejection_fraction,
            'anaemia': anaemia,
            'creatinine_phosphokinase': creatinine_phosphokinase,
            'diabetes': diabetes,
            'high_blood_pressure': high_blood_pressure,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium,
            'sex': sex,
            'platelets': platelets,
            'smoking': smoking,
            'time': time}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header('DATA UJI')
st.write(df)
st.write('---')

st.header('HASIL PREDIKSI')
input_data_as_numpy_array = np.asarray(df)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0]== 0):
  st.write("Hidup")
if (prediction[0]== 1):
 st.write("Meninggal")
