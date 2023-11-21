# stroke prediction
In this work, we aimed to predict the likelihood of a stroke using a dataset. We used machine learning techniques, specifically the Random Forest Classifier and Support Vector Machine (SVM), to analyze and predict strokes. The dataset included various factors like age, gender, lifestyle, and health-related information.

We first prepared the data by converting categorical information into numerical values and handling missing data such as unknown smoking status or unavailable body mass index (BMI). Then, we split the dataset into training, validation, and test sets.

Initially, we experimented with SVM and Random Forest models. We evaluated the models using performance metrics like accuracy, recall, precision, and F1-score to measure their predictive capabilities. Random Forest showed better results initially.

We conducted several analyses, such as observing that some health conditions might have a stronger correlation with strokes. By refining the model and adjusting hyperparameters, we managed to improve the accuracy of stroke predictions.

Ultimately, the finalized model, based on Random Forest, removed certain columns like 'ever_married,' 'Residence_type,' 'hypertension,' and 'heart_disease.' This resulted in a more accurate prediction of strokes, achieving a higher level of performance
