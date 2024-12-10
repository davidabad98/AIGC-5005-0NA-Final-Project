# AIGC-5005-0NA-Final-Project
## Gold Price Regression
## Authors: David Abad, Rizvan Nahif, Navpreet Kaur Dusanje, Darshil Shah

### Description

This project aims to predict the daily closing price of gold (gold close) using a Hybrid RNN + Dense Network. The model leverages temporal structures in financial and economic time-series data, integrating both sequential and static features for enhanced accuracy. By preprocessing historical data on financial indices, commodity prices, and economic indicators (e.g., CPI, GDP), the project seeks to deliver reliable predictions and provide insights into gold price movements.

### Project Structure

ml_gold_price_regerssion/  
│  
├── data/                              # Data storage (e.g., raw, interim, and processed data)  
│   ├── saved_model/                   # The trained LogisticRegressionModel  
│   └── raw/                           # Original raw data (if available)  
│  
├── src/                               # Core source code  
│   ├── __init__.py                    # Makes 'src' a package  
│   ├── data_preprocessing.py          # Data cleaning and preprocessing functions  
│   ├── feature_engineering.py         # Feature selection and transformation functions  
│   ├── model.py                       # Custom Logistic Regression model code  
│   ├── train_model.py                 # Code for training the model  
│   └── shared_functions.py            # Shared functions accross the project source code  
│  
├── notebooks/                         # Jupyter notebooks for experimentation and exploration  
│   ├── Data_Analysis.ipynb            # Data analysis  
│   └── initial_exploration.ipynb      # Project exploration to develop the model  
│  
├── scripts/                           # Scripts for automation and running the pipeline   
│   └── data_pipeline.py               # Script to run data preprocessing and feature engineering  
│
├── config.yaml                        # Configuration file for data paths and parameters   
├── main.py                            # Train the CustomLogistiRegression model  
├── requirements.txt                   # Python dependencies  
└── README.md                          # Project overview, how to run the code, and more  

### Execution Steps  
> <b>Train Model</b>  

1. Run main.py file  
`python main.py`  

2. Trained model will be saved in path specified in config.yaml (_/data/saved_model_)  

3. Model Evaluation is run on main.py, evaluation metrics will be printed

### Important Considerations  

* Review requirements.txt for installing dependencies