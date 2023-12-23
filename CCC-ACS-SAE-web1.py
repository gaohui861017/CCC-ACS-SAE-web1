import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


with open('xgb_plan1.dat', 'rb') as f:
    model = pickle.load(f)

st.title('CCC-ACS-SAE In-hospital Major Serious Adverse Events Calculator')
st.subheader('Estimates admission to in-hospital major serious adverse events (mortality, shock, cardiac arrest) for patients with acute coronary syndrome.')
st.divider()
st.header('Pearls/Pitfalls')
st.write('The CCC-ACS-MSAE Score is a prospectively studied scoring system to risk stratifiy patients with diagnosed ACS to estimate their in-hospital major serious adverse events. The development of CCC-ACS-MSAE Score based on CCC-ACS project which included over 100,000 ACS patients and integrates various machine learning algorithms.')
st.header('When to use')
st.write('Patients with known acute coronary syndrome, to determine in-hospital major serious adverse events (mortality, shock, cardiac arrest) risk.')
st.header('Why use')
st.write('Many guidelines recommend more aggressive medical management for patients with a high in-hospital major serious adverse events risk (or even early invasive management for these patients). Knowing a patient’s risk early may help with management and prognostication/goals of care discussions with patient and family.')
st.write('A patient with detailed admission information can be more objectively risk stratified for their prognosis, quantify their risk, and potentially lead to shorter hospital stays, fewer inappropriate interventions, and more appropriate interventions.')
st.divider()
age_input = st.selectbox('Age,years',('<55', '[55,65)', '[65,75)','≥75'))
gender_input = st.selectbox('Gender',('man', 'woman'))
historyOfDiabetesMellitus_input = st.selectbox('History Of Diabetes Mellitus',('no', 'yes'))
historyOfIschemicStroke_input = st.selectbox('History Of Ischemic Stroke',('no', 'yes'))
historyOfRenalInsufficiency_input = st.selectbox('History Of Renal Insufficiency',('no', 'yes'))
heartRate_input = st.selectbox('Heart rate',('<60','[60,100)', '≥100'))
#historyOfMyocardialInfarction_input = st.selectbox('History Of Myocardial Infarction',('no', 'yes'))
#historyOfHeartFailure_input = st.selectbox('History Of Heart Failure',('no', 'yes'))
#historyOfCOPD_input = st.selectbox('History Of COPD',('no', 'yes'))
elevatedckmb_input = st.selectbox('Elevated CKMB',('no','yes'))
shockIndex_input = st.selectbox('Shock index',('<0.4', '[0.4,0.8)','≥0.8'))
acuteFailureOnAdmission_input = st.selectbox('Acute heart failure On Admission',('no', 'yes'))
cardiacArrestOnAdmission_input = st.selectbox('Cardiac Arrest On Admission',('no', 'yes'))
stSegmentChange_input = st.selectbox('ST Segment Change',('no', 'yes'))
#killip_input = st.selectbox('Killip',('Ⅰ', 'Ⅱ', 'Ⅲ','Ⅳ'))


SEX = np.where(gender_input=='man',0,1)
MHDM = np.where(historyOfDiabetesMellitus_input=='no',0,1)
#MHMI = np.where(historyOfMyocardialInfarction_input=='no',0,1)
#MHHF = np.where(historyOfHeartFailure_input=='no',0,1)
zuzhong = np.where(historyOfIschemicStroke_input=='no',0,1)
#MHCOPD = np.where(historyOfCOPD_input=='no',0,1)	
MHKF = np.where(historyOfRenalInsufficiency_input=='no',0,1)
ele_ckmb = np.where(elevatedckmb_input=='no',0,1)
MAHF = np.where(acuteFailureOnAdmission_input=='no',0,1)
MACA = np.where(cardiacArrestOnAdmission_input=='no',0,1)
stc = np.where(stSegmentChange_input=='no',0,1)
age_1 = np.where(age_input=='<55',1,0)
age_2 = np.where(age_input=='[55,65)',1,0)
age_3 = np.where(age_input=='[65,75)',1,0)
age_4 = np.where(age_input=='≥75',1,0)
si_1 = np.where(shockIndex_input=='<0.4',1,0)
si_2 = np.where(shockIndex_input=='[0.4,0.8)',1,0)
si_3 = np.where(shockIndex_input=='≥0.8',1,0)
bmp_1 = np.where(heartRate_input=='<60',1,0)
bmp_2 = np.where(heartRate_input=='[60,100)',1,0)
bmp_3 = np.where(heartRate_input=='≥100',1,0)
#killip_1 = np.where(killip_input=='Ⅰ',1,0)
#killip_2 = np.where(killip_input=='Ⅱ',1,0)
#killip_3 = np.where(killip_input=='Ⅲ',1,0)
#killip_4 = np.where(killip_input=='Ⅳ',1,0)

#features = np.array([SEX,MHDM,MHMI,MHHF,zuzhong,MHCOPD,MHKF,MAHF,MACA,stc,age_1,age_2,age_3,age_4,si_1,si_2,si_3,bmp_1,bmp_2,bmp_3]).reshape(1,-1)

features = np.array([SEX,MHDM,zuzhong,MHKF,ele_ckmb,MAHF,MACA,stc,age_1,age_2,age_3,age_4,si_1,si_2,si_3,bmp_1,bmp_2,bmp_3]).reshape(1,-1)

if st.button('Predict'):
    col1, col2 = st.columns(2)
    p = model.predict_proba(features)[:,1]*100
    col1.metric("Score", int(p), )
    #col2.metric("Probability of death from admission to 6 months", int(p), )
    #prediction = model.predict_proba(features)[:,1]
    #st.write(' Based on feature values, your risk score is : '+ str(int(prediction * 100)))