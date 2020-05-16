# disaster_response
Data Scientist Nanodegree

# Disaster Response Pipeline Project


1. There are two csv data file and following program will load, clean, transform and merge dataframes. After that result data frame will be written to database. You can use following command to trigger proces.

Command:
    
    `cd data/ && python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
        
Result:     
    
    ```
        Loading data...
            MESSAGES: disaster_messages.csv
            CATEGORIES: disaster_categories.csv
        Result dataset row count: 26386
        Cleaning data...
        There are 170 duplicated message
        Saving data...
            DATABASE: DisasterResponse.db
        Cleaned data saved to database!
     ```
        
    - To run ML pipeline that trains classifier and saves
     
     Command:
     
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
     Sample Output:
     
     ```
       
                           precision    recall  f1-score   support

               related       0.82      0.44      0.58      1403
               request       0.00      0.00      0.00        32
                 offer       0.79      0.58      0.67      3285
           aid_related       0.65      0.19      0.30       622
          medical_help       0.73      0.24      0.36       398
      medical_products       0.74      0.14      0.23       223
     search_and_rescue       0.00      0.00      0.00       152
              security       0.67      0.23      0.34       248
              military       0.00      0.00      0.00         0
           child_alone       0.74      0.67      0.70       538
                 water       0.81      0.72      0.76       912
                  food       0.79      0.52      0.62       696
               shelter       0.72      0.48      0.58       121
              clothing       0.63      0.21      0.31       198
                  cold       0.67      0.05      0.10       418
                  ....
         other_weather       0.78      0.34      0.47      1550

           avg / total       0.75      0.42      0.51     19066
        Saving model...
        
        Overall Accuracy: 0.44641495041952706

            MODEL: models/classifier.pkl
        Trained model saved!
      ```


2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
dakcicek$ python run.py
/Users/dakcicek/opt/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/__init__.py:15: FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
  warnings.warn(msg, category=FutureWarning)
 * Serving Flask app "run" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://localhost:3001/ (Press CTRL+C to quit)
 * Restarting with fsevents reloader
/Users/dakcicek/opt/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/__init__.py:15: FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
  warnings.warn(msg, category=FutureWarning)
 * Debugger is active!
 * Debugger PIN: 308-548-945
127.0.0.1 - - [16/May/2020 04:16:41] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [16/May/2020 04:16:43] "GET /favicon.ico HTTP/1.1" 404 -



### Web App :

Go : http://localhost:3001/


![home page](https://github.com/dakcicek/disaster_response/blob/master/app/screen-1.png)
![query text](https://github.com/dakcicek/disaster_response/blob/master/app/screen-2.png)
![visualizations](https://github.com/dakcicek/disaster_response/blob/master/app/screen-3.png)
