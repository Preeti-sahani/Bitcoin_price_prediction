import streamlit as st
import pickle

st.title("Bitcoin Prediction")

pickle_in = open('model_pkl', 'rb')
lr = pickle.load(pickle_in)

number1 = st.number_input('btc_market_cap', key='1')
number2 = st.number_input('btc_miners_revenue', key='2')
number3 = st.number_input('btc_estimated_transaction_volume_rate', key='3')
number4 = st.number_input('btc_hash_rate', key='4')
number5 = st.number_input('btc_difficulty', key='5')
number6 = st.number_input('btc_trade_volume', key='6')
number7 = st.number_input('btc_cost_per_transaction', key='7')


if st.button("Predict"):
    pred = str(lr.predict([[number1, number2, number3, number4, number5, number6, number7]]))
    st.success("Price_prediction : " + pred)


