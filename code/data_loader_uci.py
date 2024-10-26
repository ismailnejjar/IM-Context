"""
IO module for train/test regression datasets
"""
import numpy as np
import pandas as pd
import os
import h5py


#####################################
# individual data files             #
#####################################
vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/uci")

def _load_boston():
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'boston-housing/boston_housing.txt'))
    X    = data[:, :-1]
    y    = data[:,  -1]
    return X, y


def _load_powerplant():
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(data_dir, 'power-plant/Folds5x2_pp.xlsx')
    data = pd.read_excel(data_file)
    x    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return x, y


def _load_concrete():
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
    data = pd.read_excel(data_file)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y


def _load_yacht():
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(data_dir, 'yacht/yacht_hydrodynamics.data')
    data = pd.read_csv(data_file, delim_whitespace=True)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y

def _load_abalone():
    data_file = os.path.join(data_dir, 'abalone/abalone.data')
    data = pd.read_csv(data_file,header = None)
    X    = data.values[:, 1:-1]
    y    = data.values[:,  -1]
    return X, y


def _load_communities():
    data_file = os.path.join(data_dir, 'communities/communities.data')
    data = pd.read_csv(data_file,header = None)
    data = data._get_numeric_data()
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y



def _load_energy_efficiency():
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(data_dir, 'energy-efficiency/ENB2012_data.xlsx')
    data      = pd.read_excel(data_file)
    X         = data.values[:, :-2]
    y_heating = data.values[:, -2]
    y_cooling = data.values[:, -1]
    return X, y_cooling


def _load_wine():
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    data_file = os.path.join(data_dir, 'wine-quality/winequality-white.csv')
    data     = pd.read_csv(data_file, sep=';')
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y

def _load_kin8nm():
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.

    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).

    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
    data     = pd.read_csv(data_file, sep=',')
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y

def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = np.loadtxt(os.path.join(data_dir, 'naval/data.txt'))
    X = data[:, :-2]
    y_compressor = data[:, -2]
    y_turbine = data[:, -1]
    return X, y_turbine

def load_dataset(name, num_bins = 10,split_seed=0, test_fraction=.2):
    # load full dataset
    load_funs = { "wine"              : _load_wine,
                  "boston"            : _load_boston,
                  "concrete"          : _load_concrete,
                  "power-plant"       : _load_powerplant,
                  "yacht"             : _load_yacht,
                  "communities"       : _load_communities,
                  "abalone"           : _load_abalone,
                  "energy-efficiency" : _load_energy_efficiency,
                  "kin8nm"            : _load_kin8nm,
                  "naval"             : _load_naval}


    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if(y.shape[0]<= 1000):
        num_bins = 15
        test_fraction = 0.1
    else:
        num_bins = 50
        test_fraction = test_fraction
        
    if(len(list(set(y)))<num_bins):
        num_bins = len(list(set(y)))

    data = pd.DataFrame(np.column_stack([X, y]), columns=[f'feature_{i}' for i in range(X.shape[1])] + ['target'])

    # Create quantile bins for the target variable
    data['bin'] = pd.cut(data['target'], bins=num_bins, duplicates='drop', labels=False)
    
    # Determine the uniform size for the test set from each bin
    min_bin_size = int(np.floor(len(data) * test_fraction / num_bins))
    # Split the data ensuring uniform test size across bins
    train_idx, test_idx = [], []

    for bin_label in range(num_bins):
        bin_data = data[data['bin'] == bin_label]
        # Ensure that the bin has enough samples for the desired test size
        if len(bin_data) > min_bin_size:
            bin_test = bin_data.sample(n=min_bin_size, random_state=split_seed)
            bin_train = bin_data.drop(bin_test.index)
        else:
            bin_test = bin_data
            bin_train = pd.DataFrame(columns=data.columns)
        train_idx.extend(bin_train.index)
        test_idx.extend(bin_test.index)
    
    # Use the indices to create training and testing sets
    X_train = data.loc[train_idx, data.columns != 'target'].values[:,:-1]
    X_test = data.loc[test_idx, data.columns != 'target'].values[:,:-1]
    y_train = data.loc[train_idx, 'target'].values
    y_test = data.loc[test_idx, 'target'].values

    y_train = np.expand_dims(y_train,axis=-1)
    y_test = np.expand_dims(y_test,axis=-1)
    
    return (X_train, y_train), (X_test, y_test)
