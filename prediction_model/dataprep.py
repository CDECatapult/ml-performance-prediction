import numpy as np
from sklearn.preprocessing import StandardScaler


def data_preprocess(dataframe,data_cols,split):
    """Function to normalise dataset and generate train/test/validation data
    Input:
        dataframe: pands dataframe object
        split: 3-element vector, adding up to 1,
            ratio of test,train,validation dataset
    """

    if not sum(split)==1:
        print("values in split must add up to 1 (is {:.3f})".format(sum(split)))

    use_target = 'timeUsed_median'

    df_data = dataframe.loc[:,data_cols].values

    scaler = StandardScaler()
    scaler.fit(df_data)

    dataScale = scaler.transform(df_data)

    dflen = dataScale.shape[0]

    test_split = int(np.floor(split[0]*dflen))
    validation_split = int(np.floor((split[0]+split[1])*dflen))
    shuffleInd = np.random.permutation(dflen)

    train, test, validation = np.split(shuffleInd,[test_split,validation_split])

    data={}
    data['Train'] = dataScale[train,:]
    data['Test'] = dataScale[test,:]
    data['Validation'] = dataScale[validation,:]

    time={}
    time['Train'] = dataframe.values[train,dataframe.columns.get_loc(use_target)].astype(np.float32)
    time['Test'] = dataframe.values[test,dataframe.columns.get_loc(use_target)].astype(np.float32)
    time['Validation'] = dataframe.values[validation,dataframe.columns.get_loc(use_target)].astype(np.float32)


    print("Size of train dataset: %d \n"
          "Size of test dataset: %d \n"
          "Size of validation dataset: %d"
          %(len(time['Train']),len(time['Test']),len(time['Validation'])))

    return data, time, train, test, validation, scaler


def data_preprocess_keep(dataframe,data_cols,split,scaler):
    """Function to normalise dataset and generate train/test/validation data
    Input:
        dataframe: pands dataframe object
        split: 3-element vector, adding up to 1,
            ratio of test,train,validation dataset
    """

    if not sum(split)==1:
        print("values in split must add up to 1 (is {:.3f})".format(sum(split)))

    use_target = 'timeUsed_median'

    df_data = dataframe.loc[:,data_cols]

    dataScale = scaler.transform(df_data.values)

    dflen = dataScale.shape[0]

    test_split = int(np.floor(split[0]*dflen))
    validation_split = int(np.floor((split[0]+split[1])*dflen))
    shuffleInd = np.random.permutation(dflen)

    train, test, validation = np.split(shuffleInd,[test_split,validation_split])

    data={}
    data['Train'] = dataScale[train,:]
    data['Test'] = dataScale[test,:]
    data['Validation'] = dataScale[validation,:]

    time={}
    time['Train'] = dataframe.values[train,dataframe.columns.get_loc(use_target)].astype(np.float32)
    time['Test'] = dataframe.values[test,dataframe.columns.get_loc(use_target)].astype(np.float32)
    time['Validation'] = dataframe.values[validation,dataframe.columns.get_loc(use_target)].astype(np.float32)


    print("Size of train dataset: %d \n"
          "Size of test dataset: %d \n"
          "Size of validation dataset: %d"
          %(len(time['Train']),len(time['Test']),len(time['Validation'])))

    return data, time, train, test, validation
