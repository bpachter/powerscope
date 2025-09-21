"""
Streamlit application for PowerScope electricity demand forecasting.
"""

import streamlit as st


def main():
    """Main Streamlit application."""
    st.title("PowerScope - Electricity Demand Forecasting")
    st.write("Forecast electricity demand with credible P10/P50/P90 bands using public ISO + weather data.")


if __name__ == "__main__":
    main()