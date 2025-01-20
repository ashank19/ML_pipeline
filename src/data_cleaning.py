def clean_training_data(data):
    """
    Perform basic data cleaning.
    """
    import pandas as pd
    # Drop duplicates
    data = data.drop_duplicates()

    # Handle missing values
    data.replace({'France':0,'Spain':1,'Germany':2},inplace=True)

    data.replace({'Male':0,'Female':1},inplace=True)

    def Normalize(X):
        X_norm = (X-X.min())/(X.max()-X.min())

        return X_norm
    #data['row_number'] = data.groupby('CustomerId')['Age'].rank(method='first', ascending=False)
    
    #df1 = data[data['row_number']==1]

    X= data[['CreditScore','Age','Balance','Tenure','NumOfProducts','EstimatedSalary']]
    x_norm = Normalize(X)

    X = pd.concat([x_norm,data[['Geography','Gender','HasCrCard','IsActiveMember','Exited']]],axis=1)


    return X

def clean_testing_data(data):
    import pandas as pd
    # Drop duplicates
    data = data.drop_duplicates()

    # Handle missing values
    data.replace({'France':0,'Spain':1,'Germany':2},inplace=True)

    data.replace({'Male':0,'Female':1},inplace=True)

    def Normalize(X):
        X_norm = (X-X.min())/(X.max()-X.min())

        return X_norm
    #data['row_number'] = data.groupby('CustomerId')['Age'].rank(method='first', ascending=False)
    
    #df1 = data[data['row_number']==1]

    X= data[['CreditScore','Age','Balance','Tenure','NumOfProducts','EstimatedSalary']]
    x_norm = Normalize(X)

    X = pd.concat([x_norm,data[['Geography','Gender','HasCrCard','IsActiveMember']]],axis=1)


    return X