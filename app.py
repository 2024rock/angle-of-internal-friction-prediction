# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import joblib

# ページの設定
st.set_page_config(
    page_title="内部摩擦角予測アプリ",
    layout="centered",
    initial_sidebar_state="expanded",
)

# タイトルと説明
st.title("内部摩擦角予測アプリ")
st.markdown("""
このアプリケーションでは、入力されたパラメータに基づいて内部摩擦角（φ）を予測します。
サイドバーから必要な数値を入力し、**予測**ボタンをクリックしてください。
""")

# サイドバーでの入力

#def user_input_features():
    #st.sidebar.header("入力パラメータ")
    #col1 = st.sidebar.columns(3)
    #col2 = st.sidebar.columns(3)
    #col3 = st.sidebar.columns(3)
    #with col1:
        #deep = st.number_input("深さ（deep）", min_value=0.0, step=0.1, format="%.2f")
        #void_ratio = st.number_input("間隙率（void ratio）", min_value=0.0, step=0.01, format="%.3f")
        #water_content = st.number_input("含水比（water content）", min_value=0.0, step=0.01, format="%.3f")

    #with col2:
        #X = st.number_input("X", min_value=-500.0, step=0.1, format="%.2f")
        #Y = st.number_input("Y", min_value=-500.0, step=0.1, format="%.2f")
        #UU = st.selectbox("UU", options=[0, 1])  # 0か1のみ選択可能

    #with col3:
        #CU = st.selectbox("CU", options=[0, 1])  # 0か1のみ選択可能
        #CUBar = st.selectbox("CUBar", options=[0, 1])  # 0か1のみ選択可能
        #CD = st.selectbox("CD", options=[0, 1])  # 0か1のみ選択可能

# CSSでサイドバーのスタイルを調整
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding: 10px;
    }
    .stNumberInput, .stSelectbox {
        padding-right: 2px;
    }
    .stNumberInput label, .stSelectbox label {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3列のサイドバー設定
def user_input_features():
    st.sidebar.header("入力パラメータ")

    # 各列に直接アクセス
    st.sidebar.write("一般")
    #col1 = st.sidebar.columns(3)
    col1 = st.sidebar.columns([1, 1, 1])  # 幅を固定するため、各列の割合を指定
    with col1[0]:
        depth = st.number_input("深さ（depth）", min_value=0.0, step=0.1, format="%.2f")
    with col1[1]:
        void_ratio = st.number_input("間隙率（void ratio）", min_value=0.0, step=0.01, format="%.3f")
    with col1[2]:
        water_content = st.number_input("自然含水比（water content）", min_value=0.0, step=0.01, format="%.3f")

    st.sidebar.write("座標（土質により決定）")
    col2 = st.sidebar.columns(2)
    with col2[0]:
        X = st.number_input("X", min_value=-500.0, step=0.1, format="%.2f")
    with col2[1]:
        Y = st.number_input("Y", min_value=-500.0, step=0.1, format="%.2f")

    st.sidebar.write("三軸圧縮試験")
    col3 = st.sidebar.columns(4)
    with col3[0]:
        UU = st.selectbox("UU", options=[0, 1])  # 0か1のみ選択可能
    with col3[1]:
        CU = st.selectbox("CU", options=[0, 1])  # 0か1のみ選択可能
    with col3[2]:
        CUBar = st.selectbox("CUBar", options=[0, 1])  # 0か1のみ選択可能
    with col3[3]:
        CD = st.selectbox("CD", options=[0, 1])  # 0か1のみ選択可能

    data = {
        'depth': depth,
        'void ratio': void_ratio,
        'water content': water_content,
        'X': X,
        'Y': Y,
        'UU': UU,
        'CU': CU,
        'CUBar': CUBar,
        'CD': CD
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ユーザー入力の表示
st.subheader("入力パラメータ")
st.write(input_df)

# モデルの読み込み
@st.cache_resource
def load_trained_model(model_path='model_tf2.h5'):
    if not os.path.exists(model_path):
        st.error(f"モデルファイル '{model_path}' が存在しません。モデルが正しく保存されていることを確認してください。")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        #model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None

model = load_trained_model()

# スケーラーの読み込み（必要な場合）
#@st.cache_resource
#def load_scaler(scaler_path='trained_model/scaler.joblib'):
    #if not os.path.exists(scaler_path):
        #st.warning(f"スケーラーファイル '{scaler_path}' が存在しません。入力データは標準化されません。")
        #return None
    #try:
        #scaler = joblib.load(scaler_path)
        #return scaler
    #except Exception as e:
        #st.error(f"スケーラーの読み込み中にエラーが発生しました: {e}")
        #return None

#scaler = load_scaler()

# 予測関数
def predict(model, input_data):
    if model is None:
        return "モデルが読み込まれていません。"
    try:
        # scalerを使わずにそのまま予測
        prediction = model.predict(input_data)
        return prediction[0][0]
    except Exception as e:
        return f"予測中にエラーが発生しました: {e}"

# 予測ボタン
if st.button("予測"):
    if model:
        # 入力データをfloat32型に変換
        input_data = input_df.values.astype(np.float32)
        prediction = predict(model, input_data)  # scalerを削除
        st.subheader("予測結果")
        st.write(f"**予測されたφ:** {prediction:.2f}")
    else:
        st.error("モデルが読み込まれていません。")

#st.write(f"TensorFlow version: {tf.__version__}")
#st.write(f"Streamlit version: {st.__version__}")
