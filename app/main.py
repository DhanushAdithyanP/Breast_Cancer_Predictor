import streamlit as st
import pandas as pd
import pickle
import numpy as np

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = pd.read_csv('Data/modified_data.csv')
    
    slider_labels = [
        ('Mean Radius', 'radius_mean'),
        ('Mean Texture', 'texture_mean'),
        ('Mean Smoothness', 'smoothness_mean'),
        ('Mean Compactness', 'compactness_mean'),
        ('Mean Symmetry', 'symmetry_mean'),
        ('Radius Standard Error', 'radius_se'),
        ('Texture Standard Error', 'texture_se'),
        ('Smoothness Standard Error', 'smoothness_se'),
        ('Compactness Standard Error', 'compactness_se'),
        ('Symmetry Standard Error', 'symmetry_se'),
        ('Fractal Dimension Standard Error', 'fractal_dimension_se')
    ]
    
    input_dict = {}
    
    for label, key in slider_labels:
        min_val = data[key].min()
        max_val = data[key].max()
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(data[key].mean())
        )
        
    return input_dict

def add_predictions(input_data):
    model = pickle.load(open("model_1.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    input_array = np.array([input_data[key] for key in input_data]).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    prob_benign = model.predict_proba(input_array_scaled)[0][0] * 100
    prob_malignant = model.predict_proba(input_array_scaled)[0][1] * 100

    st.markdown("""
    <div style='background-color: rgba(0, 0, 0, 2); padding: 20px; border: 2px solid #ccc; border-radius: 10px; text-align: center; color: white;'>
        <h2>The cell cluster is:</h2>
        <h3 style='color: {};'>{}</h3>
        <p>Probability of being benign: {:.2f}%</p>
        <p>Probability of being malignant: {:.2f}%</p>
    </div>
    """.format(
        'lightgreen' if prediction[0] == 0 else 'red',
        'Benign' if prediction[0] == 0 else 'Malignant',
        prob_benign,
        prob_malignant
    ), unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()
    
    column_info = {
        'recall_score': {
            'image_path': '../Plots/3.png'
        },
        'mean_radius': {
            'image_path': '../Plots/1.png'
        },
        'jointplot': {
            'image_path': '../Plots/4.png'
        },
        'violinplot': {
            'image_path': '../Plots/5.png'
        }
    }

    # Streamlit app layout
    st.markdown("<h1 style='text-align: center;'>Breast Cancer Predictor</h1>", unsafe_allow_html=True)
    
    add_predictions(input_data)
    
    st.markdown("<h2 style='text-align: center;'>Insights</h2>", unsafe_allow_html=True)
    
    # Display information for each column
    for key, info in column_info.items():
        st.image(info['image_path'], use_column_width=True)
        st.write("")  # Add some space between sections if needed
    
if __name__ == '__main__':
    main()