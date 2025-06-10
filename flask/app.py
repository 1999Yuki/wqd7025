import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 加载模型
model = joblib.load('best_random_forest_model.joblib')

# 特征定义
numeric_features = [
    'Size', 'bdm', 'btm', 'entertainment_count',
    'Density people / km²', 'Distance to Nearest CBD (km)', 'Distance to Singapore (km)'
]
boolean_features = [
    'Parking Facility', 'swimming pool facility', 'Lift facility'
]
property_type_columns = [
    'Property Type_Apartment / Condominium', 'Property Type_House'
]
property_desc_columns = [
    'Property_description_1-storey Terraced House',
    'Property_description_1.5-storey Terraced House',
    'Property_description_2-storey Terraced House',
    'Property_description_2.5-storey Terraced House',
    'Property_description_3-storey Terraced House',
    'Property_description_Apartment',
    'Property_description_Bungalow House',
    'Property_description_Cluster House',
    'Property_description_Condominium',
    'Property_description_Duplex',
    'Property_description_Flat',
    'Property_description_Link Bungalow',
    'Property_description_Others',
    'Property_description_Semi-Detached House',
    'Property_description_Service Residence',
    'Property_description_Studio',
    'Property_description_Terraced House',
    'Property_description_Townhouse',
    'Property_description_Townhouse Condo'
]
location_columns = [
    'Location_Batu Pahat','Location_Gelang Patah','Location_Horizon Hills',
    'Location_Iskandar Puteri','Location_Johor Bahru','Location_Kluang',
    'Location_Kota Tinggi','Location_Kulai','Location_Masai','Location_Muar',
    'Location_Pasir Gudang','Location_Pengerang','Location_Perling',
    'Location_Permas Jaya','Location_Pontian','Location_Segamat','Location_Senai',
    'Location_Setia Indah','Location_Setia Tropika','Location_Skudai','Location_Tampoi',
    'Location_Tangkak','Location_Tebrau','Location_Ulu Tiram','Location_other'
]

FEATURES = numeric_features + boolean_features + property_type_columns + property_desc_columns + location_columns

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    # 默认值
    default_values = {
        'Size': 770,
        'bdm': 3,
        'btm': 1,
        'entertainment_count': 48.0,
        'Density people / km²': 5293.94,
        'Distance to Nearest CBD (km)': 32.97,
        'Distance to Singapore (km)': 52.107,
        'Parking Facility': 0,
        'swimming pool facility': 0,
        'Lift facility': 0,
        'property_type': 'Property Type_House',
        'property_description': 'Property_description_2-storey Terraced House',
        'location': 'Location_Kulai'
    }

    if request.method == 'POST':
        data = {}
        # 获取表单数据
        for feat in numeric_features:
            data[feat] = float(request.form[feat])
        for feat in boolean_features:
            data[feat] = int(request.form.get(feat, 0))
        sel_type = request.form['property_type']
        for col in property_type_columns:
            data[col] = 1 if col == sel_type else 0
        sel_desc = request.form['property_description']
        for col in property_desc_columns:
            data[col] = 1 if col == sel_desc else 0
        sel_loc = request.form['location']
        for col in location_columns:
            data[col] = 1 if col == sel_loc else 0

        X = pd.DataFrame([data])[model.feature_names_in_]
        y_log = model.predict(X)
        y_pred = np.exp(y_log)
        prediction = round(float(y_pred[0]), 2)

        # 用于保留表单回填
        default_values.update(request.form)

    return render_template('index.html',
                           numeric_features=numeric_features,
                           boolean_features=boolean_features,
                           property_type_columns=property_type_columns,
                           property_desc_columns=property_desc_columns,
                           location_columns=location_columns,
                           prediction=prediction,
                           defaults=default_values)

if __name__ == '__main__':
    app.run(debug=True)
