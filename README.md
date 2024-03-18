
Energy Efficiency Prediction using Polynomial Regression
Project Overview
This project explores the application of Polynomial Regression to predict energy efficiency in buildings, utilizing the ENB2012_data.csv dataset. Our objective is to predict two target variables related to energy efficiency, demonstrating the model's capability to capture complex non-linear relationships between the features and the target variables.

Dataset
The dataset, ENB2012_data.csv, consists of various building parameters that could influence a building's energy efficiency, such as structural characteristics and environmental factors. The last two columns of the dataset represent the target variables we aim to predict, which quantify the building's energy efficiency.

Methodology
Data Preprocessing: The dataset is initially loaded, with features and target variables separated for model training and testing.
Polynomial Feature Transformation: We transform the input features into polynomial features of degree 2 to capture the non-linear relationships.
Model Training and Prediction: A Polynomial Regression model is trained on the transformed features. Predictions are then made for the test dataset.
Evaluation: The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics.
Results
The Polynomial Regression model achieved an R-squared (R²) value of 0.97, indicating an excellent fit to the data. This high level of accuracy suggests that the model can predict the energy efficiency of buildings with great precision.

Visualization
The project includes visualization of the model's predictions against the true values for both target variables, providing a clear visual representation of the model's predictive performance.

Installation and Usage
Requirements
Python 3.x
Pandas
NumPy
Matplotlib
scikit-learn
Running the Project
To run this project, ensure you have all the necessary Python packages installed. Execute the script Second_training_project.py with the ENB2012_data.csv dataset in the same directory. The script will perform the data preprocessing, model training, prediction, evaluation, and visualization steps.

Contribution
Contributions to this project are welcome. You can contribute by improving the model, suggesting new features, or enhancing the data visualization aspects. Please follow the standard GitHub fork-and-pull request workflow.

Acknowledgments
This project is inspired by the pursuit of understanding energy efficiency in buildings through advanced regression techniques.
