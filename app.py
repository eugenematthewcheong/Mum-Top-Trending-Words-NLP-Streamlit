import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

#For Project 2
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
model_filepath = os.path.join(__location__, "lasso.pkl")
scaler_filepath = os.path.join(__location__, "scaler.pkl")

# load the train model
with open(model_filepath, 'rb') as mf:
    model = pickle.load(mf)

with open(scaler_filepath, 'rb') as sf:
    scalerfile = pickle.load(sf)

def main():
    style = """<div style='background-color:grey; padding:12px'>
              <h1 style='color:black'>House Price Evaluator</h1>
              <p style='color:black'>Developed by: Pius Yee | Lim Zheng Gang | Eugene Matthew Cheong</p>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))

    input_planning_area = left.selectbox('Select preferred planning area', ('Ang Mo Kio', 'Bedok', 'Bishan', 'Bukit Batok', 'Bukit Merah', 'Bukit Panjang', 'Bukit Timah', 'Changi', 'Choa Chu Kang','Clementi', 'Downtown Core', 'Geylang', 'Hougang', 'Jurong East', 'Jurong West', 'Kallang', 'Marine Parade', 'Novena', 'Outram', 'Pasir Ris', 'Punggol', 'Queenstown', 'Rochor', 'Sembawang', 'Sengkang','Serangoon', 'Tampines', 'Tanglin', 'Toa Payoh', 'Western Water Catchment', 'Woodlands', 'Yishun'), index=11)
    input_flat_type = right.selectbox('Select Flat Type',('1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'), index=3)
    input_is_premium = left.selectbox('Improved-Masionette or DBSS?', ('No', 'Yes'))
    input_is_terrace = left.selectbox('Terrace type?', ('No', 'Yes'))
    input_is_superlargeterrace = right.selectbox('Terrace and larger than 2200 square feet?', ('No', 'Yes'))
    input_is_pre_war = left.selectbox('Pre-War type?', ('No', 'Yes'))
    input_floor_area_sqft = left.slider('Enter preferred floor area (square feet)',  step=1.0, format='%.2f',min_value=0.0, max_value=5000.0, value= 900.0)
    input_mid = right.slider('Enter preferred floor in the block', step =1,format="%f",min_value=1, max_value=50, value=3)
    input_max_floor_lvl = left.number_input('Preferred maximum floor of the block', step=1.0, format='%.1f', value=20.0)
    input_tenure = right.slider('Enter preferred tenure', step =1,format="%f",min_value=1, max_value=99, value=19)
    input_transaction_year = left.slider('Input Transaction Year', step =1,format="%f",min_value=2012, max_value=2021, value=2021)
    input_mrt_nearest_distance = right.slider('Preferred distance to MRT (metres)',  step=1.0, format='%.2f',min_value=0.0, max_value=10000.0, value= 3000.0)
    input_mh = left.slider('Enter preferred distance to Mall (metres)', step=1.0, format='%.2f',min_value=1.0, max_value=4000.0, value= 100.0)
    input_from_centre_distance = st.slider('Preferred distance from CBD (metres)',  step=1.0, format='%.2f',min_value=0.0, max_value=35000.0, value= 7200.0)

    button = st.button('Predict')
    # if button is pressed
    if button:
        # make prediction
        result = predict(input_planning_area,input_flat_type,input_is_premium,input_is_terrace,input_is_pre_war,input_is_superlargeterrace,input_floor_area_sqft,input_mid,input_max_floor_lvl,input_tenure,input_transaction_year,input_mrt_nearest_distance,input_mh,input_from_centre_distance)

        st.success(f'The predicted evaluation value is ${result}')


def process_mh(input_mh):
    #converts to KM
    return input_mh/1000

def process_planning_area(input_planning_area):
    list_planning_area = ['Kallang', 'Bishan', 'Bukit Batok', 'Yishun', 'Geylang', 'Hougang',
       'Bedok', 'Sengkang', 'Tampines', 'Serangoon', 'Bukit Merah',
       'Bukit Panjang', 'Woodlands', 'Jurong West', 'Toa Payoh',
       'Choa Chu Kang', 'Sembawang', 'Novena', 'Ang Mo Kio', 'Pasir Ris',
       'Clementi', 'Punggol', 'Jurong East', 'Rochor', 'Queenstown',
       'Bukit Timah', 'Outram', 'Tanglin', 'Marine Parade',
       'Western Water Catchment', 'Downtown Core', 'Changi']  

    selected_planning_area = input_planning_area

    area_category_mapping = {'Group1': ['Tanglin', 'Bukit Timah', 'Outram','Downtown Core','Bishan'],
                             'GroupJB': ['Bukit Merah','Jurong East'],
                             'GroupCQS': ['Queenstown','Serangoon','Clementi'],
                             'GroupCM': ['Marine Parade','Changi'],
                             'GroupYH': ['Hougang','Yishun'],
                             'GroupPWC': ['Bukit Panjang','Choa Chu Kang','Woodlands'],
                             'Group2': ['Western Water Catchment'],
                             'GroupA': ['Ang Mo Kio']
                             }
    
    selected_category_results = {}
    for group, area in area_category_mapping.items():
        if selected_planning_area in area:
            selected_category_results[group] = 1
        else:
            selected_category_results[group] = 0


    mature_list = ["Ang Mo Kio","Bedok","Bishan","Bukit Merah","Bukit Timah","Clementi","Downtown Core","Geylang","Kallang","Marine Parade", "Novena", "Outram", "Pasir Ris", "Queenstown", "Rochor","Serangoon","Tampines","Tanglin","Toa Payoh"] 

    if selected_planning_area in mature_list:
        selected_category_results['mature'] = 1
    else:
        selected_category_results['mature'] = 0

    
    return selected_category_results


def process_tenure_buckets(input_tenure):
    if input_tenure in range(0,11):
        return 1
    else:
        return 0


def process_year_category(input_transaction_year):
    group_year_dict = {'Group1': [2015,2016,2018],
                       'Group2': [2014,2017,2020],
                       'Group0': [2019]
                       }
    
    
    selected_year_results = {}
    for group, year in group_year_dict.items():
        if input_transaction_year in year:
            selected_year_results[group] = 1
        else:
            selected_year_results[group] = 0
    
    return selected_year_results


def process_choice(input_choice):
    if input_choice == "Yes":
        return 1
    else:
        return 0


def process_flat_type(input_flat_type):
    flat_type_dict = {}
    if input_flat_type == '1 ROOM': #1 Room
        flat_type_dict['1 ROOM'] = 1
        flat_type_dict['2 ROOM'] = 0
    elif input_flat_type == '2 ROOM':
        flat_type_dict['1 ROOM'] = 0
        flat_type_dict['2 ROOM'] = 1
    else:
        flat_type_dict['1 ROOM'] = 0
        flat_type_dict['2 ROOM'] = 0
    
    return flat_type_dict





def predict(input_planning_area,input_flat_type,input_is_premium,input_is_terrace,input_is_pre_war,input_is_superlargeterrace,input_floor_area_sqft,input_mid,input_max_floor_lvl,input_tenure,input_transaction_year,input_mrt_nearest_distance,input_mh,input_from_centre_distance):
    # processing user input

    mid = input_mid
    max_floor_lvl = input_max_floor_lvl
    mrt_nearest_distance = input_mrt_nearest_distance
    tenure = input_tenure
    mh = process_mh(input_mh)
    
    processed_planning_area = process_planning_area(input_planning_area)
    planning_area_category_Group1 = processed_planning_area['Group1'] #Returns boolean
    planning_area_category_GroupCM = processed_planning_area['GroupCM'] #Returns boolean
    planning_area_category_GroupPWC = processed_planning_area['GroupPWC'] #Returns boolean
    planning_area_category_GroupYH = processed_planning_area['GroupYH'] #Returns boolean
    planning_area_category_GroupCQS = processed_planning_area['GroupCQS'] #Returns boolean
    planning_area_category_GroupJB = processed_planning_area['GroupJB'] #Returns boolean
    planning_area_category_GroupA = processed_planning_area['GroupA'] #Returns boolean
    mature = processed_planning_area['mature'] #Returns boolean

    tenure_buckets_0_10 = process_tenure_buckets(input_tenure) #Returns boolean

    from_centre_distance = input_from_centre_distance/1000

    processed_year_category = process_year_category(input_transaction_year) #Returns a dictionary
    year_category_Group2 = processed_year_category['Group2']
    year_category_Group1 = processed_year_category['Group1']
    year_category_Group0 = processed_year_category['Group0']

    is_premium = process_choice(input_is_premium)
    is_terrace = process_choice(input_is_terrace)
    is_pre_war = process_choice(input_is_pre_war)
    is_superlargeterrace = process_choice(input_is_superlargeterrace)

    processed_flat_type = process_flat_type(input_flat_type) #Returns a dictionary
    flat_type_1_ROOM = processed_flat_type['1 ROOM']
    flat_type_2_ROOM = processed_flat_type['2 ROOM']

    floor_area_sqft = input_floor_area_sqft

    required_model_inputs= [mid, floor_area_sqft, max_floor_lvl, mrt_nearest_distance, is_pre_war, mature, year_category_Group1, planning_area_category_GroupA, planning_area_category_GroupCM, planning_area_category_GroupCQS, planning_area_category_GroupJB, planning_area_category_GroupPWC,planning_area_category_GroupYH, tenure, tenure_buckets_0_10, year_category_Group0, year_category_Group1, year_category_Group2, is_premium, is_terrace, is_superlargeterrace, flat_type_1_ROOM, flat_type_2_ROOM, from_centre_distance, mh]
    


    df = pd.DataFrame(required_model_inputs).transpose()
    df.columns = ['mid', 'floor_area_sqft', 'max_floor_lvl', 'mrt_nearest_distance',
       'is_pre_war', 'mature', 'planning_area_category_Group1',
       'planning_area_category_GroupA', 'planning_area_category_GroupCM',
       'planning_area_category_GroupCQS', 'planning_area_category_GroupJB',
       'planning_area_category_GroupPWC', 'planning_area_category_GroupYH',
       'tenure', 'tenure_buckets_0-10', 'year_category_Group0',
       'year_category_Group1', 'year_category_Group2', 'is_premium',
       'is_terrace', 'is_superlargeterrace', 'flat_type_1 ROOM',
       'flat_type_2 ROOM', 'from_centre_distance', 'mh']
    
    
    X_scaled = scalerfile.transform(df)
    y_pred = model.predict(X_scaled)
    predict_value = np.exp(y_pred)-1
    result = str(format(predict_value[0],'.2f'))
    result = result.replace("[","").replace("]","")
    return result



if __name__ == '__main__':
    main()



