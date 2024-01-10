import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time


def main():
    data = pd.read_pickle("pilot_train_svm.pkl")
    data = data.drop_duplicates(subset=['query', 'phrase'], keep='first')
    chosen_data = data[['cosine', 'entailment score', 'contradiction score', 'neutral score', 'relevance labels']]

    relevant_labels = (chosen_data['relevance labels']).sum()
    irrelevant_labels = chosen_data.shape[0] - relevant_labels
    fraction_relevant_labels = round(relevant_labels/chosen_data.shape[0], 3)

    training_set, test_set = train_test_split(chosen_data, test_size=0.3, random_state=1)
    X_train = training_set.iloc[:, 0:4].values
    Y_train = training_set.iloc[:, 4].values
    X_test = test_set.iloc[:, 0:4].values
    Y_test = test_set.iloc[:, 4].values

    # train the model on train set
    model = SVC()
    model.fit(X_train, Y_train)

    # print prediction results
    print("Before grid search:")
    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))

    print()

    # Record the start time
    start_time = time.time()
    print('rbf')
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(Y_test, grid_predictions))
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")

    # Record the start time
    start_time = time.time()
    print("\n")
    print('sigmoid')
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['sigmoid']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(Y_test, grid_predictions))
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")


    # Record the start time
    start_time = time.time()
    print("\n")
    print('linear')
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'kernel': ['linear']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(Y_test, grid_predictions))
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")



    # Record the start time
    start_time = time.time()
    print("\n")
    print('poly')
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'degree': [2, 3, 4],
                  'kernel': ['poly']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(Y_test, grid_predictions))
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")


if __name__ == "__main__":
    main()