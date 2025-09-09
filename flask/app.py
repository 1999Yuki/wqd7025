import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 加载模型（确保文件存在）
model = joblib.load('best_random_forest_model.joblib')

# 如果训练时 target=log(price)，保持 True；否则改为 False
log_transformed = True

# ========= 写死：location → 三个联动数值（由你的 CSV 生成）=========
LOCATION_DEFAULTS = {
    'Location_Kulai': {'Density people / km²': 5293.94, 'Distance to Nearest CBD (km)': 32.97, 'Distance to Singapore (km)': 52.107},
    'Location_Johor Bahru': {'Density people / km²': 23809.92, 'Distance to Nearest CBD (km)': 0.0, 'Distance to Singapore (km)': 19.687},
    'Location_Iskandar Puteri': {'Density people / km²': 1009.1, 'Distance to Nearest CBD (km)': 0.0, 'Distance to Singapore (km)': 40.154},
    'Location_Permas Jaya': {'Density people / km²': 2222.63, 'Distance to Nearest CBD (km)': 11.235, 'Distance to Singapore (km)': 27.492},
    'Location_Skudai': {'Density people / km²': 5053.24, 'Distance to Nearest CBD (km)': 16.467, 'Distance to Singapore (km)': 34.644},
    'Location_Taman Mount Austin': {'Density people / km²': 8926.54, 'Distance to Nearest CBD (km)': 15.532, 'Distance to Singapore (km)': 31.719},
    'Location_Taman Molek': {'Density people / km²': 10610.46, 'Distance to Nearest CBD (km)': 10.373, 'Distance to Singapore (km)': 26.6},
    'Location_Taman Setia Tropika': {'Density people / km²': 3121.55, 'Distance to Nearest CBD (km)': 18.923, 'Distance to Singapore (km)': 36.908},
    'Location_Taman Sutera Utama': {'Density people / km²': 7512.67, 'Distance to Nearest CBD (km)': 15.549, 'Distance to Singapore (km)': 32.789},
    'Location_Taman Ungku Tun Aminah': {'Density people / km²': 10366.23, 'Distance to Nearest CBD (km)': 15.704, 'Distance to Singapore (km)': 33.054},
    'Location_Bukit Indah': {'Density people / km²': 3775.29, 'Distance to Nearest CBD (km)': 19.86, 'Distance to Singapore (km)': 38.257},
    'Location_Pasir Gudang': {'Density people / km²': 1816.34, 'Distance to Nearest CBD (km)': 20.303, 'Distance to Singapore (km)': 28.346},
    'Location_Tebrau': {'Density people / km²': 4876.4, 'Distance to Nearest CBD (km)': 12.948, 'Distance to Singapore (km)': 29.311},
    'Location_Taman Pelangi': {'Density people / km²': 20361.6, 'Distance to Nearest CBD (km)': 2.858, 'Distance to Singapore (km)': 23.362},
    'Location_Taman Abad': {'Density people / km²': 16428.82, 'Distance to Nearest CBD (km)': 2.783, 'Distance to Singapore (km)': 23.041},
    'Location_Taman Sentosa': {'Density people / km²': 12593.38, 'Distance to Nearest CBD (km)': 3.084, 'Distance to Singapore (km)': 22.846},
    'Location_Larkin': {'Density people / km²': 8282.88, 'Distance to Nearest CBD (km)': 5.074, 'Distance to Singapore (km)': 26.079},
    'Location_Taman Adda Heights': {'Density people / km²': 3977.75, 'Distance to Nearest CBD (km)': 10.816, 'Distance to Singapore (km)': 29.977},
    'Location_Masai': {'Density people / km²': 2946.47, 'Distance to Nearest CBD (km)': 19.834, 'Distance to Singapore (km)': 27.787},
    'Location_Taman Johor Jaya': {'Density people / km²': 6871.3, 'Distance to Nearest CBD (km)': 13.634, 'Distance to Singapore (km)': 30.071},
    'Location_Taman Daya': {'Density people / km²': 5965.79, 'Distance to Nearest CBD (km)': 13.669, 'Distance to Singapore (km)': 30.183},
    'Location_Kota Tinggi': {'Density people / km²': 49.2, 'Distance to Nearest CBD (km)': 41.351, 'Distance to Singapore (km)': 65.567},
    'Location_Gelang Patah': {'Density people / km²': 1346.48, 'Distance to Nearest CBD (km)': 23.035, 'Distance to Singapore (km)': 40.392},
    'Location_Horizon Hills': {'Density people / km²': 1968.73, 'Distance to Nearest CBD (km)': 20.888, 'Distance to Singapore (km)': 38.558},
    'Location_Setia Indah': {'Density people / km²': 5798.31, 'Distance to Nearest CBD (km)': 12.989, 'Distance to Singapore (km)': 30.372},
    'Location_Permas Jaya (Jaya)': {'Density people / km²': 2222.63, 'Distance to Nearest CBD (km)': 11.235, 'Distance to Singapore (km)': 27.492},
    'Location_Ulu Tiram': {'Density people / km²': 594.21, 'Distance to Nearest CBD (km)': 23.612, 'Distance to Singapore (km)': 39.651},
    'Location_Senai': {'Density people / km²': 2172.51, 'Distance to Nearest CBD (km)': 23.849, 'Distance to Singapore (km)': 42.33},
    'Location_Mount Austin': {'Density people / km²': 8926.54, 'Distance to Nearest CBD (km)': 15.532, 'Distance to Singapore (km)': 31.719},
    'Location_Tampoi': {'Density people / km²': 8619.11, 'Distance to Nearest CBD (km)': 9.477, 'Distance to Singapore (km)': 28.626},
    'Location_Taman Desa Cemerlang': {'Density people / km²': 4702.53, 'Distance to Nearest CBD (km)': 17.738, 'Distance to Singapore (km)': 33.869},
    'Location_Taman Desa Tebrau': {'Density people / km²': 4920.2, 'Distance to Nearest CBD (km)': 12.311, 'Distance to Singapore (km)': 29.247},
    'Location_Taman Mutiara Rini': {'Density people / km²': 4652.0, 'Distance to Nearest CBD (km)': 18.02, 'Distance to Singapore (km)': 35.156},
    'Location_Taman Perling': {'Density people / km²': 5754.3, 'Distance to Nearest CBD (km)': 10.878, 'Distance to Singapore (km)': 28.655},
    'Location_Perling': {'Density people / km²': 5754.3, 'Distance to Nearest CBD (km)': 10.878, 'Distance to Singapore (km)': 28.655},
    'Location_Pontian': {'Density people / km²': 168.22, 'Distance to Nearest CBD (km)': 57.095, 'Distance to Singapore (km)': 73.655},
    'Location_Segamat': {'Density people / km²': 205.72, 'Distance to Nearest CBD (km)': 170.286, 'Distance to Singapore (km)': 195.439},
    'Location_Batu Pahat': {'Density people / km²': 319.07, 'Distance to Nearest CBD (km)': 97.074, 'Distance to Singapore (km)': 121.49},
    'Location_Muar': {'Density people / km²': 214.6, 'Distance to Nearest CBD (km)': 150.321, 'Distance to Singapore (km)': 174.951},
    'Location_Tangkak': {'Density people / km²': 143.57, 'Distance to Nearest CBD (km)': 168.236, 'Distance to Singapore (km)': 192.455},
    'Location_Pengerang': {'Density people / km²': 83.14, 'Distance to Nearest CBD (km)': 92.566, 'Distance to Singapore (km)': 76.046},
    'Location_Tebrau (Johor Bahru)': {'Density people / km²': 4876.4, 'Distance to Nearest CBD (km)': 12.948, 'Distance to Singapore (km)': 29.311},
    'Location_Taman Setia Indah': {'Density people / km²': 5798.31, 'Distance to Nearest CBD (km)': 12.989, 'Distance to Singapore (km)': 30.372},
    'Location_Taman Austin Heights': {'Density people / km²': 8926.54, 'Distance to Nearest CBD (km)': 15.532, 'Distance to Singapore (km)': 31.719},
    'Location_Pasir Gudang (Johor Bahru)': {'Density people / km²': 1816.34, 'Distance to Nearest CBD (km)': 20.303, 'Distance to Singapore (km)': 28.346},
    'Location_Kluang': {'Density people / km²': 262.64, 'Distance to Nearest CBD (km)': 97.263, 'Distance to Singapore (km)': 121.355},
    'Location_Seri Alam': {'Density people / km²': 2946.47, 'Distance to Nearest CBD (km)': 21.767, 'Distance to Singapore (km)': 29.722},
    'Location_other': {'Density people / km²': 2946.47, 'Distance to Nearest CBD (km)': 21.767, 'Distance to Singapore (km)': 29.722}
}

# ===== 特征定义 =====
numeric_features = [
    'Size', 'bdm', 'btm', 'entertainment_count',
    'Density people / km²', 'Distance to Nearest CBD (km)', 'Distance to Singapore (km)'
]
boolean_features = ['Parking Facility', 'swimming pool facility', 'Lift facility']
property_type_columns = ['Property Type_Apartment / Condominium', 'Property Type_House']
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
        'Size': 750,
        'bdm': 3,
        'btm': 1,
        'entertainment_count': 101.0,
        'Density people / km²': 23809.92,
        'Distance to Nearest CBD (km)': 0.00,
        'Distance to Singapore (km)': 19.687,
        'Parking Facility': 1,
        'swimming pool facility': 0,
        'Lift facility': 0,
        'property_type': 'Property Type_Apartment / Condominium',
        'property_description': 'Property_description_Flat',
        'location': 'Location_Johor Bahru'
    }

    # 如果默认 location 在映射里，用映射覆盖三项
    if default_values['location'] in LOCATION_DEFAULTS:
        for k, v in LOCATION_DEFAULTS[default_values['location']].items():
            default_values[k] = v

    form_values = default_values.copy()

    if request.method == 'POST':
        try:
            data = {}

            # 获取 location
            sel_loc = request.form['location']
            form_values['location'] = sel_loc

            # 获取数值特征（先按表单值读取）
            for feat in numeric_features:
                value = float(request.form[feat])
                data[feat] = value
                form_values[feat] = value

            # 如果该 location 在映射里，用映射覆盖三项数值
            if sel_loc in LOCATION_DEFAULTS:
                for k, v in LOCATION_DEFAULTS[sel_loc].items():
                    data[k] = float(v)
                    form_values[k] = float(v)

            # 获取布尔特征
            for feat in boolean_features:
                value = int(request.form.get(feat, 0))
                data[feat] = value
                form_values[feat] = value

            # Property type one-hot 编码
            sel_type = request.form['property_type']
            for col in property_type_columns:
                data[col] = 1 if col == sel_type else 0
            form_values['property_type'] = sel_type

            # Property description one-hot 编码
            sel_desc = request.form['property_description']
            for col in property_desc_columns:
                data[col] = 1 if col == sel_desc else 0
            form_values['property_description'] = sel_desc

            # Location one-hot 编码
            for col in location_columns:
                data[col] = 1 if col == sel_loc else 0

            # 构建 DataFrame，预测（列顺序对齐训练时的 feature_names_in_）
            X = pd.DataFrame([data])[model.feature_names_in_]
            y_output = model.predict(X)
            y_pred = np.exp(y_output) if log_transformed else y_output
            prediction = round(float(y_pred[0]), 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        'index.html',
        numeric_features=numeric_features,
        boolean_features=boolean_features,
        property_type_columns=property_type_columns,
        property_desc_columns=property_desc_columns,
        location_columns=location_columns,
        prediction=prediction,
        defaults=form_values,
        location_defaults=LOCATION_DEFAULTS  # ✅ 传给前端用于联动
    )


if __name__ == '__main__':
    app.run(debug=True)
