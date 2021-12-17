import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


def preprocess():
    # Load files
    train_df = pd.read_csv('dataset\\Training Data.csv')
    test_df = pd.read_csv('dataset\\Test Data.csv')
    # General info about the df
    print(train_df.info())
    print(train_df.head())
    print(train_df.describe())
    # Drop columns
    train_df = train_df.drop('Id', axis=1)  # column wise
    # Show histogram
    hist = train_df.hist()
    plt.show()
    # Correlation matrix as heatmap
    correlation = train_df.corr()
    corr = sns.heatmap(correlation, vmin=-1, vmax=1, annot=True, cmap='seismic')
    plt.show()
    # Drop even more columns
    train_df = train_df.drop('STATE', axis=1)
    train_df = train_df.drop('CITY', axis=1)
    train_df = train_df.drop('Profession', axis=1)
    # Repeat for test df
    test_df = test_df.drop('Id', axis=1)
    test_df = test_df.drop('STATE', axis=1)
    test_df = test_df.drop('CITY', axis=1)
    test_df = test_df.drop('Profession', axis=1)
    # Encode variables
    car_map = {'yes': 1, 'no': 0}
    train_df['Car_Ownership'] = train_df['Car_Ownership'].map(car_map)
    married_map = {'married': 1, 'single': 0}
    train_df['Married/Single'] = train_df['Married/Single'].map(married_map)
    # Same goes for test df
    test_df['Car_Ownership'] = test_df['Car_Ownership'].map(car_map)
    test_df['Married/Single'] = test_df['Married/Single'].map(married_map)
    # Ordinal encoding
    house_encoder = OrdinalEncoder(categories=[['norent_noown', 'rented', 'owned']])
    train_df['House_Ownership'] = house_encoder.fit_transform(train_df['House_Ownership'].values.reshape(-1, 1))
    test_df['House_Ownership'] = house_encoder.fit_transform(test_df['House_Ownership'].values.reshape(-1, 1))
    # X and Y
    x_train = train_df.drop('Risk_Flag', axis=1)
    y_train = train_df['Risk_Flag']
    # As usual with test df
    x_test = test_df.drop('Risk_Flag', axis=1)
    y_test = test_df['Risk_Flag']
    return x_train, y_train, x_test, y_test
