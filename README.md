# Car Price Prediction

A machine learning project to predict used car prices based on various features such as year, fuel type, transmission, and seller type. Helps buyers and sellers estimate a fair price quickly.

---

## Problem Statement

Estimating the price of a used car is often challenging due to many influencing factors. This project builds a regression model that provides accurate price predictions, helping users make informed decisions.

---

## Dataset

The dataset contains features such as car model, year, selling price, fuel type, transmission, and seller type. It includes over 8,000 records sourced from publicly available car sales data.

---

## Features

- **Year:** Manufacturing year of the car  
- **Fuel Type:** Petrol, Diesel, CNG, etc.  
- **Seller Type:** Individual or Dealer  
- **Transmission:** Manual or Automatic  
- **Selling Price:** Target variable (in INR)

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/Samru2589/Car-Price-Prediction.git
cd Car-Price-Prediction
# Install dependencies
pip install -r requirements.txt

## Project Structure
- data/                # Dataset CSV files
- notebooks/           #  notebooks (.py )
- src/                 # Python scripts
- results/             # Model outputs and visualizations
- requirements.txt     # Python dependencies
- README.md            # Project documentation

## Model Details
Used Linear Regression and Random Forest Regressor
Achieved R² score of 0.91 on test data
RMSE: 15,000 INR

## License
This project is licensed under the MIT License.

## Sample Result
<!-- ![Model Performance](images/performance_plot.png) -->

