

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

/kaggle/input/house-prices-advanced-regression-techniques/train.csv
/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
/kaggle/input/house-prices-advanced-regression-techniques/test.csv
/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.shape

(1460, 81)

test.shape

(1459, 80)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
PoolQC 	1453 	0.995205
MiscFeature 	1406 	0.963014
Alley 	1369 	0.937671
Fence 	1179 	0.807534
FireplaceQu 	690 	0.472603
LotFrontage 	259 	0.177397
GarageCond 	81 	0.055479
GarageType 	81 	0.055479
GarageYrBlt 	81 	0.055479
GarageFinish 	81 	0.055479
GarageQual 	81 	0.055479
BsmtExposure 	38 	0.026027
BsmtFinType2 	38 	0.026027
BsmtFinType1 	37 	0.025342
BsmtCond 	37 	0.025342
BsmtQual 	37 	0.025342
MasVnrArea 	8 	0.005479
MasVnrType 	8 	0.005479
Electrical 	1 	0.000685
Utilities 	0 	0.000000

total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
PoolQC 	1456 	0.997944
MiscFeature 	1408 	0.965045
Alley 	1352 	0.926662
Fence 	1169 	0.801234
FireplaceQu 	730 	0.500343
LotFrontage 	227 	0.155586
GarageCond 	78 	0.053461
GarageQual 	78 	0.053461
GarageYrBlt 	78 	0.053461
GarageFinish 	78 	0.053461
GarageType 	76 	0.052090
BsmtCond 	45 	0.030843
BsmtQual 	44 	0.030158
BsmtExposure 	44 	0.030158
BsmtFinType1 	42 	0.028787
BsmtFinType2 	42 	0.028787
MasVnrType 	16 	0.010966
MasVnrArea 	15 	0.010281
MSZoning 	4 	0.002742
BsmtHalfBath 	2 	0.001371

train.isnull().sum()

Id                 0
MSSubClass         0
MSZoning           0
LotFrontage      259
LotArea            0
                ... 
MoSold             0
YrSold             0
SaleType           0
SaleCondition      0
SalePrice          0
Length: 81, dtype: int64

train['PoolQC'].isnull().sum()

1453

train.drop('PoolQC', axis=1, inplace=True)
test.drop('PoolQC', axis=1, inplace=True)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
MiscFeature 	1406 	0.963014
Alley 	1369 	0.937671
Fence 	1179 	0.807534
FireplaceQu 	690 	0.472603
LotFrontage 	259 	0.177397
GarageCond 	81 	0.055479
GarageType 	81 	0.055479
GarageYrBlt 	81 	0.055479
GarageFinish 	81 	0.055479
GarageQual 	81 	0.055479
BsmtExposure 	38 	0.026027
BsmtFinType2 	38 	0.026027
BsmtFinType1 	37 	0.025342
BsmtCond 	37 	0.025342
BsmtQual 	37 	0.025342
MasVnrArea 	8 	0.005479
MasVnrType 	8 	0.005479
Electrical 	1 	0.000685
Utilities 	0 	0.000000
YearRemodAdd 	0 	0.000000

train.drop('MiscFeature', axis=1, inplace=True)
test.drop('MiscFeature', axis=1, inplace=True)

train['Alley'].replace(np.NAN,"No_allay",inplace = True)
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	NaN 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	NaN 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NaN 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NaN 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NaN 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

train['Fence'].replace(np.NAN,"No_Fence",inplace = True)
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
FireplaceQu 	690 	0.472603
LotFrontage 	259 	0.177397
GarageCond 	81 	0.055479
GarageType 	81 	0.055479
GarageYrBlt 	81 	0.055479
GarageFinish 	81 	0.055479
GarageQual 	81 	0.055479
BsmtExposure 	38 	0.026027
BsmtFinType2 	38 	0.026027
BsmtFinType1 	37 	0.025342
BsmtCond 	37 	0.025342
BsmtQual 	37 	0.025342
MasVnrType 	8 	0.005479
MasVnrArea 	8 	0.005479
Electrical 	1 	0.000685
Exterior2nd 	0 	0.000000
OverallCond 	0 	0.000000
Exterior1st 	0 	0.000000
RoofMatl 	0 	0.000000
ExterQual 	0 	0.000000

#BsmtFinType2
train['BsmtFinType2'].replace(np.NAN,"No_Base",inplace = True)
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

train['FireplaceQu'].replace(np.NAN,"NoFire",inplace = True)
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

train['GarageCond'].replace(np.NAN,"NoGar",inplace = True)
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	No_allay 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	No_allay 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	No_Fence 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

test['GarageCond'].replace(np.NAN,"NoGar",inplace = True)
test['FireplaceQu'].replace(np.NAN,"NoFire",inplace = True)
test['BsmtFinType2'].replace(np.NAN,"NoBas",inplace = True)
test['Fence'].replace(np.NAN,"NNoFen",inplace = True)
test['Alley'].replace(np.NAN,"NoAl",inplace = True)
train['BsmtFinType2'].replace("No_Base","NoBas",inplace = True)
train['Fence'].replace("No_Fence","NoFen",inplace = True)
train['Alley'].replace("No_allay","NoAl",inplace = True)

train['GarageType'].replace(np.NAN,"NoGar",inplace = True)
train['GarageQual'].replace(np.NAN,"NoGar",inplace = True)
train['GarageFinish'].replace(np.NAN,"NoGar",inplace = True)
train['BsmtExposure'].replace(np.NAN,"NoBas",inplace = True)
train['BsmtFinType1'].replace(np.NAN,"NoBas",inplace = True)
train['BsmtCond'].replace(np.NAN,"NoBas",inplace = True)
train['BsmtQual'].replace(np.NAN,"NoBas",inplace = True)
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].dropna().median())
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].dropna().mode().values[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].dropna().mode().values[0])
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].dropna().median())

test['GarageType'].replace(np.NAN,"NoGar",inplace = True)
test['GarageQual'].replace(np.NAN,"NoGar",inplace = True)
test['GarageFinish'].replace(np.NAN,"NoGar",inplace = True)
test['BsmtExposure'].replace(np.NAN,"NoBas",inplace = True)
test['BsmtFinType1'].replace(np.NAN,"NoBas",inplace = True)
test['BsmtCond'].replace(np.NAN,"NoBas",inplace = True)
test['BsmtQual'].replace(np.NAN,"NoBas",inplace = True)
test['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].dropna().median())
test['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].dropna().mode().values[0])
test['Electrical'] = train['Electrical'].fillna(train['Electrical'].dropna().mode().values[0])
test['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].dropna().median())

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
GarageYrBlt 	81 	0.055479
YearBuilt 	0 	0.000000
YearRemodAdd 	0 	0.000000
RoofStyle 	0 	0.000000
RoofMatl 	0 	0.000000
Exterior1st 	0 	0.000000
Exterior2nd 	0 	0.000000
MasVnrType 	0 	0.000000
MasVnrArea 	0 	0.000000
ExterQual 	0 	0.000000
SalePrice 	0 	0.000000
Foundation 	0 	0.000000
BsmtQual 	0 	0.000000
BsmtCond 	0 	0.000000
BsmtExposure 	0 	0.000000
BsmtFinType1 	0 	0.000000
BsmtFinSF1 	0 	0.000000
BsmtFinType2 	0 	0.000000
BsmtFinSF2 	0 	0.000000
ExterCond 	0 	0.000000

train.describe()

	Id 	MSSubClass 	LotFrontage 	LotArea 	OverallQual 	OverallCond 	YearBuilt 	YearRemodAdd 	MasVnrArea 	BsmtFinSF1 	... 	WoodDeckSF 	OpenPorchSF 	EnclosedPorch 	3SsnPorch 	ScreenPorch 	PoolArea 	MiscVal 	MoSold 	YrSold 	SalePrice
count 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	... 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000 	1460.000000
mean 	730.500000 	56.897260 	69.863699 	10516.828082 	6.099315 	5.575342 	1971.267808 	1984.865753 	103.117123 	443.639726 	... 	94.244521 	46.660274 	21.954110 	3.409589 	15.060959 	2.758904 	43.489041 	6.321918 	2007.815753 	180921.195890
std 	421.610009 	42.300571 	22.027677 	9981.264932 	1.382997 	1.112799 	30.202904 	20.645407 	180.731373 	456.098091 	... 	125.338794 	66.256028 	61.119149 	29.317331 	55.757415 	40.177307 	496.123024 	2.703626 	1.328095 	79442.502883
min 	1.000000 	20.000000 	21.000000 	1300.000000 	1.000000 	1.000000 	1872.000000 	1950.000000 	0.000000 	0.000000 	... 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	1.000000 	2006.000000 	34900.000000
25% 	365.750000 	20.000000 	60.000000 	7553.500000 	5.000000 	5.000000 	1954.000000 	1967.000000 	0.000000 	0.000000 	... 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	5.000000 	2007.000000 	129975.000000
50% 	730.500000 	50.000000 	69.000000 	9478.500000 	6.000000 	5.000000 	1973.000000 	1994.000000 	0.000000 	383.500000 	... 	0.000000 	25.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	6.000000 	2008.000000 	163000.000000
75% 	1095.250000 	70.000000 	79.000000 	11601.500000 	7.000000 	6.000000 	2000.000000 	2004.000000 	164.250000 	712.250000 	... 	168.000000 	68.000000 	0.000000 	0.000000 	0.000000 	0.000000 	0.000000 	8.000000 	2009.000000 	214000.000000
max 	1460.000000 	190.000000 	313.000000 	215245.000000 	10.000000 	9.000000 	2010.000000 	2010.000000 	1600.000000 	5644.000000 	... 	857.000000 	547.000000 	552.000000 	508.000000 	480.000000 	738.000000 	15500.000000 	12.000000 	2010.000000 	755000.000000

8 rows × 38 columns

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
GarageYrBlt 	81 	0.055479
YearBuilt 	0 	0.000000
YearRemodAdd 	0 	0.000000
RoofStyle 	0 	0.000000
RoofMatl 	0 	0.000000
Exterior1st 	0 	0.000000
Exterior2nd 	0 	0.000000
MasVnrType 	0 	0.000000
MasVnrArea 	0 	0.000000
ExterQual 	0 	0.000000
SalePrice 	0 	0.000000
Foundation 	0 	0.000000
BsmtQual 	0 	0.000000
BsmtCond 	0 	0.000000
BsmtExposure 	0 	0.000000
BsmtFinType1 	0 	0.000000
BsmtFinSF1 	0 	0.000000
BsmtFinType2 	0 	0.000000
BsmtFinSF2 	0 	0.000000
ExterCond 	0 	0.000000

train['GarageYrBlt'].replace(np.NAN,0,inplace = True)
train['GarageYrBlt'] = train['GarageYrBlt'].astype(int)
train['GarageYrBlt'].unique()

array([2003, 1976, 2001, 1998, 2000, 1993, 2004, 1973, 1931, 1939, 1965,
       2005, 1962, 2006, 1960, 1991, 1970, 1967, 1958, 1930, 2002, 1968,
       2007, 2008, 1957, 1920, 1966, 1959, 1995, 1954, 1953,    0, 1983,
       1977, 1997, 1985, 1963, 1981, 1964, 1999, 1935, 1990, 1945, 1987,
       1989, 1915, 1956, 1948, 1974, 2009, 1950, 1961, 1921, 1900, 1979,
       1951, 1969, 1936, 1975, 1971, 1923, 1984, 1926, 1955, 1986, 1988,
       1916, 1932, 1972, 1918, 1980, 1924, 1996, 1940, 1949, 1994, 1910,
       1978, 1982, 1992, 1925, 1941, 2010, 1927, 1947, 1937, 1942, 1938,
       1952, 1928, 1922, 1934, 1906, 1914, 1946, 1908, 1929, 1933])

 

train['GarageYrBlt'].describe()

count    1460.000000
mean     1868.739726
std       453.697295
min         0.000000
25%      1958.000000
50%      1977.000000
75%      2001.000000
max      2010.000000
Name: GarageYrBlt, dtype: float64

import random

train['GarageYrBlt'].replace(np.NAN,random.randint(1900,2010),inplace = True)

train['GarageYrBlt'].describe()

count    1460.000000
mean     1868.739726
std       453.697295
min         0.000000
25%      1958.000000
50%      1977.000000
75%      2001.000000
max      2010.000000
Name: GarageYrBlt, dtype: float64

test['GarageYrBlt'].describe()

count    1381.000000
mean     1977.721217
std        26.431175
min      1895.000000
25%      1959.000000
50%      1979.000000
75%      2002.000000
max      2207.000000
Name: GarageYrBlt, dtype: float64

test['GarageYrBlt'].max()

2207.0

test['GarageYrBlt'].replace(np.NAN,random.randint(1895.0,2207.0),inplace = True)
test['GarageYrBlt'] = test['GarageYrBlt'].astype(int)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
SalePrice 	0 	0.0
ExterCond 	0 	0.0
RoofStyle 	0 	0.0
RoofMatl 	0 	0.0
Exterior1st 	0 	0.0
Exterior2nd 	0 	0.0
MasVnrType 	0 	0.0
MasVnrArea 	0 	0.0
ExterQual 	0 	0.0
Foundation 	0 	0.0
YearBuilt 	0 	0.0
BsmtQual 	0 	0.0
BsmtCond 	0 	0.0
BsmtExposure 	0 	0.0
BsmtFinType1 	0 	0.0
BsmtFinSF1 	0 	0.0
BsmtFinType2 	0 	0.0
BsmtFinSF2 	0 	0.0
YearRemodAdd 	0 	0.0
OverallCond 	0 	0.0

total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
MSZoning 	4 	0.002742
Functional 	2 	0.001371
BsmtFullBath 	2 	0.001371
BsmtHalfBath 	2 	0.001371
Utilities 	2 	0.001371
GarageArea 	1 	0.000685
BsmtFinSF2 	1 	0.000685
BsmtUnfSF 	1 	0.000685
SaleType 	1 	0.000685
Exterior2nd 	1 	0.000685
Exterior1st 	1 	0.000685
KitchenQual 	1 	0.000685
GarageCars 	1 	0.000685
TotalBsmtSF 	1 	0.000685
BsmtFinSF1 	1 	0.000685
Neighborhood 	0 	0.000000
BsmtExposure 	0 	0.000000
MSSubClass 	0 	0.000000
LotFrontage 	0 	0.000000
LotArea 	0 	0.000000

test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].dropna().mode().values[0])
test['Functional']=test['Functional'].fillna(test['Functional'].dropna().mode().values[0])
test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].dropna().mode().values[0])
test['Utilities']=test['Utilities'].fillna(test['Utilities'].dropna().mode().values[0])
test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].dropna().mode().values[0])
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].dropna().median())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].dropna().median())
test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].dropna().mode().values[0])
test['SaleType']=test['SaleType'].fillna(test['SaleType'].dropna().mode().values[0])
test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].dropna().mode().values[0])
test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].dropna().mode().values[0])
test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].dropna().mode().values[0])
test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].dropna().mode().values[0])
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].dropna().median())
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].dropna().median())

total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1)
missing_data.head(20)

	0 	1
SaleCondition 	0 	0.0
ExterCond 	0 	0.0
RoofStyle 	0 	0.0
RoofMatl 	0 	0.0
Exterior1st 	0 	0.0
Exterior2nd 	0 	0.0
MasVnrType 	0 	0.0
MasVnrArea 	0 	0.0
ExterQual 	0 	0.0
Foundation 	0 	0.0
YearBuilt 	0 	0.0
BsmtQual 	0 	0.0
BsmtCond 	0 	0.0
BsmtExposure 	0 	0.0
BsmtFinType1 	0 	0.0
BsmtFinSF1 	0 	0.0
BsmtFinType2 	0 	0.0
BsmtFinSF2 	0 	0.0
YearRemodAdd 	0 	0.0
OverallCond 	0 	0.0

test['Functional'].unique()

array(['Typ', 'Min2', 'Min1', 'Mod', 'Maj1', 'Sev', 'Maj2'], dtype=object)

import seaborn as sns
sns.heatmap(train.corr())

<matplotlib.axes._subplots.AxesSubplot at 0x7f4b94cf2510>

train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	RL 	65.0 	8450 	Pave 	NoAl 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	NoFen 	0 	2 	2008 	WD 	Normal 	208500
1 	2 	20 	RL 	80.0 	9600 	Pave 	NoAl 	Reg 	Lvl 	AllPub 	... 	0 	0 	0 	NoFen 	0 	5 	2007 	WD 	Normal 	181500
2 	3 	60 	RL 	68.0 	11250 	Pave 	NoAl 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NoFen 	0 	9 	2008 	WD 	Normal 	223500
3 	4 	70 	RL 	60.0 	9550 	Pave 	NoAl 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NoFen 	0 	2 	2006 	WD 	Abnorml 	140000
4 	5 	60 	RL 	84.0 	14260 	Pave 	NoAl 	IR1 	Lvl 	AllPub 	... 	0 	0 	0 	NoFen 	0 	12 	2008 	WD 	Normal 	250000

5 rows × 79 columns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

# Bad metric
plt.hist(train['Street']);

ncols = 3
nrows = int(np.ceil(len(train.columns) / (1.0*ncols)))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(60, 60))

# Lazy counter so we can remove unwated axes
counter = 0
for i in range(nrows):
    for j in range(ncols):

        ax = axes[i][j]

        # Plot when we have data
        if counter < len(train.columns):

            ax.hist(train[train.columns[counter]],bins = 10, color='blue', alpha=0.5, label='{}'.format(train.columns[counter]))
            ax.set_xlabel('x')
            ax.set_ylabel('PDF')
            leg = ax.legend(loc='upper left')
            leg.draw_frame(False)

        # Remove axis when we no longer have data
        else:
            ax.set_axis_off()

        counter += 1

plt.show()

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
train['MSZoning']=number.fit_transform(train['MSZoning'].astype('str'))
train['Street']=number.fit_transform(train['Street'].astype('str'))
train['Alley']=number.fit_transform(train['Alley'].astype('str'))
train['LotShape']=number.fit_transform(train['LotShape'].astype('str'))
train['LandContour']=number.fit_transform(train['LandContour'].astype('str'))
train['Utilities']=number.fit_transform(train['Utilities'].astype('str'))
train['SaleCondition']=number.fit_transform(train['SaleCondition'].astype('str'))
train['ExterCond']=number.fit_transform(train['ExterCond'].astype('str'))
train['RoofStyle']=number.fit_transform(train['RoofStyle'].astype('str'))
train['RoofMatl']=number.fit_transform(train['RoofMatl'].astype('str'))
train['LandSlope']=number.fit_transform(train['LandSlope'].astype('str'))
train['Neighborhood']=number.fit_transform(train['Neighborhood'].astype('str'))
train['Condition1']=number.fit_transform(train['Condition1'].astype('str'))
train['Functional']=number.fit_transform(train['Functional'].astype('str'))
train['Condition2']=number.fit_transform(train['Condition2'].astype('str'))
train['FireplaceQu']=number.fit_transform(train['FireplaceQu'].astype('str'))
train.head()

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	3SsnPorch 	ScreenPorch 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice
0 	1 	60 	3 	65.0 	8450 	1 	1 	3 	3 	0 	... 	0 	0 	0 	NoFen 	0 	2 	2008 	WD 	4 	208500
1 	2 	20 	3 	80.0 	9600 	1 	1 	3 	3 	0 	... 	0 	0 	0 	NoFen 	0 	5 	2007 	WD 	4 	181500
2 	3 	60 	3 	68.0 	11250 	1 	1 	0 	3 	0 	... 	0 	0 	0 	NoFen 	0 	9 	2008 	WD 	4 	223500
3 	4 	70 	3 	60.0 	9550 	1 	1 	0 	3 	0 	... 	0 	0 	0 	NoFen 	0 	2 	2006 	WD 	0 	140000
4 	5 	60 	3 	84.0 	14260 	1 	1 	0 	3 	0 	... 	0 	0 	0 	NoFen 	0 	12 	2008 	WD 	4 	250000

5 rows × 79 columns

train.select_dtypes(include ='object') 

	LotConfig 	BldgType 	HouseStyle 	Exterior1st 	Exterior2nd 	MasVnrType 	ExterQual 	Foundation 	BsmtQual 	BsmtCond 	... 	CentralAir 	Electrical 	KitchenQual 	GarageType 	GarageFinish 	GarageQual 	GarageCond 	PavedDrive 	Fence 	SaleType
0 	Inside 	1Fam 	2Story 	VinylSd 	VinylSd 	BrkFace 	Gd 	PConc 	Gd 	TA 	... 	Y 	SBrkr 	Gd 	Attchd 	RFn 	TA 	TA 	Y 	NoFen 	WD
1 	FR2 	1Fam 	1Story 	MetalSd 	MetalSd 	None 	TA 	CBlock 	Gd 	TA 	... 	Y 	SBrkr 	TA 	Attchd 	RFn 	TA 	TA 	Y 	NoFen 	WD
2 	Inside 	1Fam 	2Story 	VinylSd 	VinylSd 	BrkFace 	Gd 	PConc 	Gd 	TA 	... 	Y 	SBrkr 	Gd 	Attchd 	RFn 	TA 	TA 	Y 	NoFen 	WD
3 	Corner 	1Fam 	2Story 	Wd Sdng 	Wd Shng 	None 	TA 	BrkTil 	TA 	Gd 	... 	Y 	SBrkr 	Gd 	Detchd 	Unf 	TA 	TA 	Y 	NoFen 	WD
4 	FR2 	1Fam 	2Story 	VinylSd 	VinylSd 	BrkFace 	Gd 	PConc 	Gd 	TA 	... 	Y 	SBrkr 	Gd 	Attchd 	RFn 	TA 	TA 	Y 	NoFen 	WD
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
1455 	Inside 	1Fam 	2Story 	VinylSd 	VinylSd 	None 	TA 	PConc 	Gd 	TA 	... 	Y 	SBrkr 	TA 	Attchd 	RFn 	TA 	TA 	Y 	NoFen 	WD
1456 	Inside 	1Fam 	1Story 	Plywood 	Plywood 	Stone 	TA 	CBlock 	Gd 	TA 	... 	Y 	SBrkr 	TA 	Attchd 	Unf 	TA 	TA 	Y 	MnPrv 	WD
1457 	Inside 	1Fam 	2Story 	CemntBd 	CmentBd 	None 	Ex 	Stone 	TA 	Gd 	... 	Y 	SBrkr 	Gd 	Attchd 	RFn 	TA 	TA 	Y 	GdPrv 	WD
1458 	Inside 	1Fam 	1Story 	MetalSd 	MetalSd 	None 	TA 	CBlock 	TA 	TA 	... 	Y 	FuseA 	Gd 	Attchd 	Unf 	TA 	TA 	Y 	NoFen 	WD
1459 	Inside 	1Fam 	1Story 	HdBoard 	HdBoard 	None 	Gd 	CBlock 	TA 	TA 	... 	Y 	SBrkr 	TA 	Attchd 	Fin 	TA 	TA 	Y 	NoFen 	WD

1460 rows × 25 columns

train['MSZoning']=number.fit_transform(train['MSZoning'].astype('str'))
train['Street']=number.fit_transform(train['Street'].astype('str'))
train['Alley']=number.fit_transform(train['Alley'].astype('str'))
train['LotShape']=number.fit_transform(train['LotShape'].astype('str'))
train['LandContour']=number.fit_transform(train['LandContour'].astype('str'))
train['Utilities']=number.fit_transform(train['Utilities'].astype('str'))
train['SaleCondition']=number.fit_transform(train['SaleCondition'].astype('str'))
train['ExterCond']=number.fit_transform(train['ExterCond'].astype('str'))
train['RoofStyle']=number.fit_transform(train['RoofStyle'].astype('str'))
train['RoofMatl']=number.fit_transform(train['RoofMatl'].astype('str'))
train['LandSlope']=number.fit_transform(train['LandSlope'].astype('str'))
train['Neighborhood']=number.fit_transform(train['Neighborhood'].astype('str'))
train['Condition1']=number.fit_transform(train['Condition1'].astype('str'))
train['Functional']=number.fit_transform(train['Functional'].astype('str'))
train['Condition2']=number.fit_transform(train['Condition2'].astype('str'))
train['FireplaceQu']=number.fit_transform(train['FireplaceQu'].astype('str'))
train['LotConfig']=number.fit_transform(train['LotConfig'].astype('str'))
train['BldgType']=number.fit_transform(train['BldgType'].astype('str'))
train['HouseStyle']=number.fit_transform(train['HouseStyle'].astype('str'))
train['Exterior1st']=number.fit_transform(train['Exterior1st'].astype('str'))
train['Exterior2nd']=number.fit_transform(train['Exterior2nd'].astype('str'))
train['MasVnrType']=number.fit_transform(train['MasVnrType'].astype('str'))
train['ExterQual']=number.fit_transform(train['ExterQual'].astype('str'))
train['Foundation']=number.fit_transform(train['Foundation'].astype('str'))
train['BsmtQual']=number.fit_transform(train['BsmtQual'].astype('str'))
train['BsmtCond']=number.fit_transform(train['BsmtCond'].astype('str'))
train['CentralAir']=number.fit_transform(train['CentralAir'].astype('str'))
train['Electrical']=number.fit_transform(train['Electrical'].astype('str'))
train['KitchenQual']=number.fit_transform(train['KitchenQual'].astype('str'))
train['GarageType']=number.fit_transform(train['GarageType'].astype('str'))
train['GarageFinish']=number.fit_transform(train['GarageFinish'].astype('str'))
train['GarageQual']=number.fit_transform(train['GarageQual'].astype('str'))
train['GarageCond']=number.fit_transform(train['GarageCond'].astype('str'))
train['PavedDrive']=number.fit_transform(train['PavedDrive'].astype('str'))
train['Fence']=number.fit_transform(train['Fence'].astype('str'))
train['SaleType']=number.fit_transform(train['SaleType'].astype('str'))
train['BsmtExposure']=number.fit_transform(train['BsmtExposure'].astype('str'))
train['BsmtFinType1']=number.fit_transform(train['BsmtFinType1'].astype('str'))
train['BsmtFinType2']=number.fit_transform(train['BsmtFinType2'].astype('str'))
train['Heating']=number.fit_transform(train['Heating'].astype('str'))
train['HeatingQC']=number.fit_transform(train['HeatingQC'].astype('str'))

train['BsmtExposure']=number.fit_transform(train['BsmtExposure'].astype('str'))
train['BsmtFinType1']=number.fit_transform(train['BsmtFinType1'].astype('str'))
train['BsmtFinType2']=number.fit_transform(train['BsmtFinType2'].astype('str'))
train['Heating']=number.fit_transform(train['Heating'].astype('str'))
train['HeatingQC']=number.fit_transform(train['HeatingQC'].astype('str'))

train.select_dtypes(include ='object') 

0
1
2
3
4
...
1455
1456
1457
1458
1459

1460 rows × 0 columns

sns.heatmap(train.corr())

<matplotlib.axes._subplots.AxesSubplot at 0x7f4ba78ffa50>

train.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 79 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   int64  
 3   LotFrontage    1460 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   int64  
 6   Alley          1460 non-null   int64  
 7   LotShape       1460 non-null   int64  
 8   LandContour    1460 non-null   int64  
 9   Utilities      1460 non-null   int64  
 10  LotConfig      1460 non-null   int64  
 11  LandSlope      1460 non-null   int64  
 12  Neighborhood   1460 non-null   int64  
 13  Condition1     1460 non-null   int64  
 14  Condition2     1460 non-null   int64  
 15  BldgType       1460 non-null   int64  
 16  HouseStyle     1460 non-null   int64  
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   int64  
 22  RoofMatl       1460 non-null   int64  
 23  Exterior1st    1460 non-null   int64  
 24  Exterior2nd    1460 non-null   int64  
 25  MasVnrType     1460 non-null   int64  
 26  MasVnrArea     1460 non-null   float64
 27  ExterQual      1460 non-null   int64  
 28  ExterCond      1460 non-null   int64  
 29  Foundation     1460 non-null   int64  
 30  BsmtQual       1460 non-null   int64  
 31  BsmtCond       1460 non-null   int64  
 32  BsmtExposure   1460 non-null   int64  
 33  BsmtFinType1   1460 non-null   int64  
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1460 non-null   int64  
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   int64  
 40  HeatingQC      1460 non-null   int64  
 41  CentralAir     1460 non-null   int64  
 42  Electrical     1460 non-null   int64  
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   int64  
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   int64  
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    1460 non-null   int64  
 58  GarageType     1460 non-null   int64  
 59  GarageYrBlt    1460 non-null   int64  
 60  GarageFinish   1460 non-null   int64  
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1460 non-null   int64  
 64  GarageCond     1460 non-null   int64  
 65  PavedDrive     1460 non-null   int64  
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  Fence          1460 non-null   int64  
 73  MiscVal        1460 non-null   int64  
 74  MoSold         1460 non-null   int64  
 75  YrSold         1460 non-null   int64  
 76  SaleType       1460 non-null   int64  
 77  SaleCondition  1460 non-null   int64  
 78  SalePrice      1460 non-null   int64  
dtypes: float64(2), int64(77)
memory usage: 901.2 KB

y = train['SalePrice']
X = train.drop('SalePrice', axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train)

/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

LogisticRegression()

ypred = model.predict(X_test)
ypred

array([240000, 140000, 100000, 205000, 151000, 147000, 143000, 140000,
       235000, 120000, 181000, 190000, 197000, 144000, 135000, 125000,
       197000, 141000, 140000, 179400, 173000, 129000, 155000, 178000,
       193000, 197900, 179400,  86000, 180000, 140000, 318000, 173000,
       125000, 239000, 318000, 160000, 207500, 147000, 178000, 239000,
       180000, 140000, 202500, 175000, 239000, 190000, 147000, 170000,
       135000, 140000, 239000, 180000, 165000, 118000, 235000, 115000,
       140000, 178000, 155000, 100000, 155000, 135000, 140000, 143000,
       207500, 147000, 135000, 135000, 125500, 135000, 173000, 125000,
       140000, 148000, 115000, 202500, 143000, 110000, 228000, 155000,
       132500, 129000, 127500, 147000, 295000, 165000, 135000, 174000,
       140000, 180000, 173000, 160000, 135000, 135000, 181000, 180000,
       230000, 240000, 110000, 174000, 180000, 178000, 140000, 173000,
       110000, 228000, 185000, 155000, 135000,  79000, 100000, 125000,
       215000, 155000, 125500, 179400, 190000, 132500, 239000, 205000,
       127500, 113000, 155000, 197000, 180000, 181000, 190000, 174000,
       153500, 143000, 207500, 174000, 189000, 143000, 115000, 155000,
       135000, 174000, 100000, 173000, 141000, 129000, 190000, 147000,
       143000, 173000, 174000, 155000, 230000, 143000, 180000, 350000,
       190000, 122000, 113000, 190000, 176000, 135000, 118000, 135000,
       140000, 143000, 178000, 155000, 318000,  85000, 140000, 250000,
       148000, 189000, 170000, 178000, 153500, 143000, 240000, 135000,
       119000, 315000, 173000, 140000, 235000, 169000, 113000, 180000,
       145000, 185000, 151000, 132500, 181000, 129000, 147000, 143000,
       202500, 140000, 140000, 125000, 129000, 144000, 135000, 148500,
       127000, 129900, 110000, 118000, 200000, 250000, 148000, 205000,
       190000, 118000, 180000, 207500, 140000, 127500, 241500, 294000,
       180000, 212000, 318000,  86000, 180000, 190000, 189000, 135000,
       181000, 235000, 205000, 190000, 127000, 140000, 140000, 154000,
       141000, 140000, 240000, 124000, 122000, 119000,  79000, 170000,
       113000, 240000, 135000, 205000, 180000, 124000, 132500, 173000,
       135000, 167000, 170000, 205000, 127500, 239000, 165000, 202500,
       127500, 275000, 625000, 184750, 179400, 178000, 135000, 112000,
       244000, 167000, 235000, 110000, 135000, 202500, 135000, 205000,
       140000, 190000, 200000, 225000, 135000, 135000, 135000, 140000,
       180000, 260000, 193000, 135000, 124000, 285000, 148500, 220000,
       244000, 193000, 127500, 140000])

evaluation = f1_score(y_test, ypred,average='macro')
evaluation

0.0008125472411186697

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)

DecisionTreeClassifier()

ypred_tree = tree.predict(X_test)
ypred_tree

array([315000, 119000, 174500, 202900, 113000, 125500, 245000,  96500,
       348000, 140000, 196000, 152000, 222500, 129000, 186700, 126500,
       215000, 132500, 159500, 119750, 124000, 120500, 110000, 260400,
       223000, 178000, 160000,  75500, 337500, 119900, 171000, 201000,
       129000, 290000, 381000, 215000, 226000, 110000, 317000, 271000,
       180000,  67000, 200141, 255500, 410000, 112000, 112500, 130000,
       160000, 112500, 335000, 175500, 179900, 116000, 235000, 110000,
       106000, 255900, 153500,  82000, 129000, 137000, 176500, 140000,
       217000, 167000, 152000, 227875,  93000, 183500, 179665, 140000,
       141000, 290000, 106500, 244600, 151500, 109500, 277000, 163500,
       135000, 133000, 116500, 119000, 137500, 162000, 122000, 171750,
       153900, 139950, 224900, 187000, 160000, 196500, 194500, 153500,
       262500, 129900, 110000, 127000, 312500, 147400, 161500, 184000,
       114504, 226000, 144000, 107000, 180000, 125500,  87000, 124500,
       260000, 163500, 165000, 148500, 374000, 126000, 226000, 230000,
       260000, 193000, 148500, 205000, 289000, 225000, 253293, 178000,
       197000, 163500, 124000, 213500, 119500, 175500, 110000, 315000,
       177000, 250580, 112500, 231500,  84500, 148000, 277000,  90000,
       241500, 201000, 239799, 150750, 197900, 235000, 190000, 212000,
       320000, 139000, 125500, 236000, 148000, 210000, 146800, 160000,
       110000, 163000, 186500, 169900, 245000, 135000, 112000, 290000,
       175500, 172400, 168500, 178000, 209500, 162500, 129500,  83000,
       139000, 229000, 103600, 175500, 190000, 242000,  73000, 372402,
       135500, 146500,  89500, 206000, 394432, 120500, 148500, 150750,
       179200, 113000,  94500, 135960,  86000, 140000, 149500,  85400,
       115000, 193000,  60000,  55000, 225000, 137500, 165000, 315500,
       240000,  94000, 155000, 278000,  90000, 118500, 190000, 229000,
       289000, 206900, 245000, 108000, 158000, 127500,  85500, 129000,
       191000, 293077, 200000, 301500, 154500, 169000,  93500, 118000,
       130000, 148000, 194500, 149000, 100000, 124500, 106000, 289000,
       113000, 231500, 192000, 335000, 215000, 145000, 224000, 227000,
       133000, 239799, 163000, 325300,  73000, 315500, 203000, 230000,
       113000, 174000, 374000, 377426, 184100, 153900, 135000, 119500,
       437154, 257500, 328000,  86000, 149500, 320000, 125500, 143500,
       125500, 199900, 130000, 253000, 174000, 125500, 215000, 143000,
       110000, 315000, 119000, 121000, 239799, 188000, 113000, 270000,
       315000, 122000,  85500, 100000])

evaluation_tree = f1_score(y_test, ypred_tree,average='macro')
evaluation_tree

0.001614434947768281

forest = RandomForestClassifier()

forest.fit(X_train,y_train)

RandomForestClassifier()

ypred_forest = forest.predict(X_test)
ypred_forest

array([223000, 105000, 169500, 178000,  94500, 119000, 230000, 125500,
       625000, 151000, 176000, 135000, 199900, 130000, 110500, 153000,
       215000,  93500, 110000, 140000, 122900, 141000, 129000, 178000,
       187000, 223500, 177000,  75500, 337500,  68400, 124000, 187000,
       146800, 290000, 253293, 185000, 266000, 125000, 222500, 290000,
       155000, 127000, 181134, 326000, 290000, 142600, 129000, 130000,
       187000, 107500, 380000, 155000, 175500,  87000, 235000, 110000,
       128000, 240000, 129000, 110000, 145000, 128900, 148000, 143000,
       217000, 161500, 129000, 233170, 135000, 215000, 178000, 141000,
        60000, 265900,  60000, 239000, 119900, 109500, 274725, 130000,
       135960, 120500, 116500, 156000, 127000, 184900, 110000, 202500,
       198900, 155000, 173000, 185000, 185000, 255000, 246578, 143000,
       194500, 135000, 110000, 149000, 226000, 116900, 139000, 149500,
       110000, 260000, 129900, 119000, 163990, 135000, 127000, 135000,
       205000, 135500, 155000, 148500, 380000, 125000, 226000, 252000,
        84500, 164700, 127500, 205000, 204000, 263435, 380000, 176000,
       197000, 157000, 140000, 206000, 118858, 223000, 110000, 207500,
       145000, 179600, 100000, 173000, 140000, 148000, 170000, 132500,
       153337, 185000, 265900, 142000, 266000, 235000, 145000, 274000,
       201000, 127500, 135000, 190000, 115000, 135000,  84500, 160000,
       110000, 148000, 186500, 145000, 245000,  85000, 140000, 315750,
       185850, 172400, 174900, 178000, 230000, 180000, 145000, 130000,
       135000, 250000, 139000, 152000, 260000, 197000, 116900, 297000,
       141000, 165150, 100000, 206000, 246578, 143000, 126000, 137500,
       190000, 134900, 135000, 132500,  86000, 140000, 140000, 125000,
       127000, 180000, 127500,  88000, 135900, 154300, 143000, 305000,
       236500,  58500, 155000, 278000,  90000,  79000, 260000, 274970,
       175500, 294000, 232000,  75500, 175000, 140000,  79000, 110000,
       191000, 290000, 191000, 215000, 132500, 165000,  93500, 143000,
        88000, 155000, 215000, 149000, 139000, 110000, 100000, 205000,
       133000, 179000, 180000, 208500, 160000, 215000, 207500, 173000,
       170000, 204000, 162000, 252000,  85000, 361919, 208300, 240000,
        86000, 174000, 320000, 184750, 174000, 135000, 135000, 100000,
       337500, 225000, 180000,  98000, 225000, 295493, 137500, 149000,
       140000, 215000, 130000, 226700, 174000, 174500, 215000,  40000,
       130000, 377426, 174500, 105000, 179600, 167000, 113000, 270000,
       284000, 127000,  52000,  79000])

evaluation_tree = f1_score(y_test, ypred_forest,average = 'macro')
evaluation_tree

0.001723356009070295

test1 = test

test1.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 78 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1459 non-null   int64  
 1   MSSubClass     1459 non-null   int64  
 2   MSZoning       1459 non-null   object 
 3   LotFrontage    1459 non-null   float64
 4   LotArea        1459 non-null   int64  
 5   Street         1459 non-null   object 
 6   Alley          1459 non-null   object 
 7   LotShape       1459 non-null   object 
 8   LandContour    1459 non-null   object 
 9   Utilities      1459 non-null   object 
 10  LotConfig      1459 non-null   object 
 11  LandSlope      1459 non-null   object 
 12  Neighborhood   1459 non-null   object 
 13  Condition1     1459 non-null   object 
 14  Condition2     1459 non-null   object 
 15  BldgType       1459 non-null   object 
 16  HouseStyle     1459 non-null   object 
 17  OverallQual    1459 non-null   int64  
 18  OverallCond    1459 non-null   int64  
 19  YearBuilt      1459 non-null   int64  
 20  YearRemodAdd   1459 non-null   int64  
 21  RoofStyle      1459 non-null   object 
 22  RoofMatl       1459 non-null   object 
 23  Exterior1st    1459 non-null   object 
 24  Exterior2nd    1459 non-null   object 
 25  MasVnrType     1459 non-null   object 
 26  MasVnrArea     1459 non-null   float64
 27  ExterQual      1459 non-null   object 
 28  ExterCond      1459 non-null   object 
 29  Foundation     1459 non-null   object 
 30  BsmtQual       1459 non-null   object 
 31  BsmtCond       1459 non-null   object 
 32  BsmtExposure   1459 non-null   object 
 33  BsmtFinType1   1459 non-null   object 
 34  BsmtFinSF1     1459 non-null   float64
 35  BsmtFinType2   1459 non-null   object 
 36  BsmtFinSF2     1459 non-null   float64
 37  BsmtUnfSF      1459 non-null   float64
 38  TotalBsmtSF    1459 non-null   float64
 39  Heating        1459 non-null   object 
 40  HeatingQC      1459 non-null   object 
 41  CentralAir     1459 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1459 non-null   int64  
 44  2ndFlrSF       1459 non-null   int64  
 45  LowQualFinSF   1459 non-null   int64  
 46  GrLivArea      1459 non-null   int64  
 47  BsmtFullBath   1459 non-null   float64
 48  BsmtHalfBath   1459 non-null   float64
 49  FullBath       1459 non-null   int64  
 50  HalfBath       1459 non-null   int64  
 51  BedroomAbvGr   1459 non-null   int64  
 52  KitchenAbvGr   1459 non-null   int64  
 53  KitchenQual    1459 non-null   object 
 54  TotRmsAbvGrd   1459 non-null   int64  
 55  Functional     1459 non-null   object 
 56  Fireplaces     1459 non-null   int64  
 57  FireplaceQu    1459 non-null   object 
 58  GarageType     1459 non-null   object 
 59  GarageYrBlt    1459 non-null   int64  
 60  GarageFinish   1459 non-null   object 
 61  GarageCars     1459 non-null   float64
 62  GarageArea     1459 non-null   float64
 63  GarageQual     1459 non-null   object 
 64  GarageCond     1459 non-null   object 
 65  PavedDrive     1459 non-null   object 
 66  WoodDeckSF     1459 non-null   int64  
 67  OpenPorchSF    1459 non-null   int64  
 68  EnclosedPorch  1459 non-null   int64  
 69  3SsnPorch      1459 non-null   int64  
 70  ScreenPorch    1459 non-null   int64  
 71  PoolArea       1459 non-null   int64  
 72  Fence          1459 non-null   object 
 73  MiscVal        1459 non-null   int64  
 74  MoSold         1459 non-null   int64  
 75  YrSold         1459 non-null   int64  
 76  SaleType       1459 non-null   object 
 77  SaleCondition  1459 non-null   object 
dtypes: float64(10), int64(27), object(41)
memory usage: 889.2+ KB

test['MSZoning']=number.fit_transform(test['MSZoning'].astype('str'))
test['Street']=number.fit_transform(test['Street'].astype('str'))
test['Alley']=number.fit_transform(test['Alley'].astype('str'))
test['LotShape']=number.fit_transform(test['LotShape'].astype('str'))
test['LandContour']=number.fit_transform(test['LandContour'].astype('str'))
test['Utilities']=number.fit_transform(test['Utilities'].astype('str'))
test['SaleCondition']=number.fit_transform(test['SaleCondition'].astype('str'))
test['ExterCond']=number.fit_transform(test['ExterCond'].astype('str'))
test['RoofStyle']=number.fit_transform(test['RoofStyle'].astype('str'))
test['RoofMatl']=number.fit_transform(test['RoofMatl'].astype('str'))
test['LandSlope']=number.fit_transform(test['LandSlope'].astype('str'))
test['Neighborhood']=number.fit_transform(test['Neighborhood'].astype('str'))
test['Condition1']=number.fit_transform(test['Condition1'].astype('str'))
test['Functional']=number.fit_transform(test['Functional'].astype('str'))
test['Condition2']=number.fit_transform(test['Condition2'].astype('str'))
test['FireplaceQu']=number.fit_transform(test['FireplaceQu'].astype('str'))
test['LotConfig']=number.fit_transform(test['LotConfig'].astype('str'))
test['BldgType']=number.fit_transform(test['BldgType'].astype('str'))
test['HouseStyle']=number.fit_transform(test['HouseStyle'].astype('str'))
test['Exterior1st']=number.fit_transform(test['Exterior1st'].astype('str'))
test['Exterior2nd']=number.fit_transform(test['Exterior2nd'].astype('str'))
test['MasVnrType']=number.fit_transform(test['MasVnrType'].astype('str'))
test['ExterQual']=number.fit_transform(test['ExterQual'].astype('str'))
test['Foundation']=number.fit_transform(test['Foundation'].astype('str'))
test['BsmtQual']=number.fit_transform(test['BsmtQual'].astype('str'))
test['BsmtCond']=number.fit_transform(test['BsmtCond'].astype('str'))
test['CentralAir']=number.fit_transform(test['CentralAir'].astype('str'))
test['Electrical']=number.fit_transform(test['Electrical'].astype('str'))
test['KitchenQual']=number.fit_transform(test['KitchenQual'].astype('str'))
test['GarageType']=number.fit_transform(test['GarageType'].astype('str'))
test['GarageFinish']=number.fit_transform(test['GarageFinish'].astype('str'))
test['GarageQual']=number.fit_transform(test['GarageQual'].astype('str'))
test['GarageCond']=number.fit_transform(test['GarageCond'].astype('str'))
test['PavedDrive']=number.fit_transform(test['PavedDrive'].astype('str'))
test['Fence']=number.fit_transform(test['Fence'].astype('str'))
test['SaleType']=number.fit_transform(test['SaleType'].astype('str'))
test['BsmtExposure']=number.fit_transform(test['BsmtExposure'].astype('str'))
test['BsmtFinType1']=number.fit_transform(test['BsmtFinType1'].astype('str'))
test['BsmtFinType2']=number.fit_transform(test['BsmtFinType2'].astype('str'))
test['Heating']=number.fit_transform(test['Heating'].astype('str'))
test['HeatingQC']=number.fit_transform(test['HeatingQC'].astype('str'))

test1.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 78 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1459 non-null   int64  
 1   MSSubClass     1459 non-null   int64  
 2   MSZoning       1459 non-null   int64  
 3   LotFrontage    1459 non-null   float64
 4   LotArea        1459 non-null   int64  
 5   Street         1459 non-null   int64  
 6   Alley          1459 non-null   int64  
 7   LotShape       1459 non-null   int64  
 8   LandContour    1459 non-null   int64  
 9   Utilities      1459 non-null   int64  
 10  LotConfig      1459 non-null   int64  
 11  LandSlope      1459 non-null   int64  
 12  Neighborhood   1459 non-null   int64  
 13  Condition1     1459 non-null   int64  
 14  Condition2     1459 non-null   int64  
 15  BldgType       1459 non-null   int64  
 16  HouseStyle     1459 non-null   int64  
 17  OverallQual    1459 non-null   int64  
 18  OverallCond    1459 non-null   int64  
 19  YearBuilt      1459 non-null   int64  
 20  YearRemodAdd   1459 non-null   int64  
 21  RoofStyle      1459 non-null   int64  
 22  RoofMatl       1459 non-null   int64  
 23  Exterior1st    1459 non-null   int64  
 24  Exterior2nd    1459 non-null   int64  
 25  MasVnrType     1459 non-null   int64  
 26  MasVnrArea     1459 non-null   float64
 27  ExterQual      1459 non-null   int64  
 28  ExterCond      1459 non-null   int64  
 29  Foundation     1459 non-null   int64  
 30  BsmtQual       1459 non-null   int64  
 31  BsmtCond       1459 non-null   int64  
 32  BsmtExposure   1459 non-null   int64  
 33  BsmtFinType1   1459 non-null   int64  
 34  BsmtFinSF1     1459 non-null   float64
 35  BsmtFinType2   1459 non-null   int64  
 36  BsmtFinSF2     1459 non-null   float64
 37  BsmtUnfSF      1459 non-null   float64
 38  TotalBsmtSF    1459 non-null   float64
 39  Heating        1459 non-null   int64  
 40  HeatingQC      1459 non-null   int64  
 41  CentralAir     1459 non-null   int64  
 42  Electrical     1459 non-null   int64  
 43  1stFlrSF       1459 non-null   int64  
 44  2ndFlrSF       1459 non-null   int64  
 45  LowQualFinSF   1459 non-null   int64  
 46  GrLivArea      1459 non-null   int64  
 47  BsmtFullBath   1459 non-null   float64
 48  BsmtHalfBath   1459 non-null   float64
 49  FullBath       1459 non-null   int64  
 50  HalfBath       1459 non-null   int64  
 51  BedroomAbvGr   1459 non-null   int64  
 52  KitchenAbvGr   1459 non-null   int64  
 53  KitchenQual    1459 non-null   int64  
 54  TotRmsAbvGrd   1459 non-null   int64  
 55  Functional     1459 non-null   int64  
 56  Fireplaces     1459 non-null   int64  
 57  FireplaceQu    1459 non-null   int64  
 58  GarageType     1459 non-null   int64  
 59  GarageYrBlt    1459 non-null   int64  
 60  GarageFinish   1459 non-null   int64  
 61  GarageCars     1459 non-null   float64
 62  GarageArea     1459 non-null   float64
 63  GarageQual     1459 non-null   int64  
 64  GarageCond     1459 non-null   int64  
 65  PavedDrive     1459 non-null   int64  
 66  WoodDeckSF     1459 non-null   int64  
 67  OpenPorchSF    1459 non-null   int64  
 68  EnclosedPorch  1459 non-null   int64  
 69  3SsnPorch      1459 non-null   int64  
 70  ScreenPorch    1459 non-null   int64  
 71  PoolArea       1459 non-null   int64  
 72  Fence          1459 non-null   int64  
 73  MiscVal        1459 non-null   int64  
 74  MoSold         1459 non-null   int64  
 75  YrSold         1459 non-null   int64  
 76  SaleType       1459 non-null   int64  
 77  SaleCondition  1459 non-null   int64  
dtypes: float64(10), int64(68)
memory usage: 889.2 KB

y_pred1 = model.predict(test1)
y_pred2 = forest.predict(test1)
y_pred3 = tree.predict(test1)

test1['SalePrice_model'] = y_pred2
test1['SalePrice_forest'] = y_pred2
test1['SalePrice_tree'] = y_pred3

test1.head(40)

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	PoolArea 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice_model 	SalePrice_forest 	SalePrice_tree
0 	1461 	20 	2 	65.0 	11622 	1 	1 	3 	3 	0 	... 	0 	2 	0 	6 	2010 	8 	4 	129000 	129000 	125500
1 	1462 	20 	3 	80.0 	14267 	1 	1 	0 	3 	0 	... 	0 	4 	12500 	6 	2010 	8 	4 	147500 	147500 	165500
2 	1463 	60 	3 	68.0 	13830 	1 	1 	0 	3 	0 	... 	0 	2 	0 	3 	2010 	8 	4 	175000 	175000 	175000
3 	1464 	60 	3 	60.0 	9978 	1 	1 	0 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	175000 	175000 	173000
4 	1465 	120 	3 	84.0 	5005 	1 	1 	0 	1 	0 	... 	0 	4 	0 	1 	2010 	8 	4 	147500 	147500 	185500
5 	1466 	60 	3 	85.0 	10000 	1 	1 	0 	3 	0 	... 	0 	4 	0 	4 	2010 	8 	4 	175000 	175000 	196500
6 	1467 	20 	3 	75.0 	7980 	1 	1 	0 	3 	0 	... 	0 	0 	500 	3 	2010 	8 	4 	210000 	210000 	187500
7 	1468 	60 	3 	69.0 	8402 	1 	1 	0 	3 	0 	... 	0 	4 	0 	5 	2010 	8 	4 	175000 	175000 	165400
8 	1469 	20 	3 	51.0 	10176 	1 	1 	3 	3 	0 	... 	0 	4 	0 	2 	2010 	8 	4 	173000 	173000 	148000
9 	1470 	20 	3 	50.0 	8400 	1 	1 	3 	3 	0 	... 	0 	2 	0 	4 	2010 	8 	4 	144000 	144000 	125500
10 	1471 	120 	2 	70.0 	5858 	1 	1 	0 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	147500 	147500 	193000
11 	1472 	160 	4 	85.0 	1680 	1 	1 	3 	3 	0 	... 	0 	4 	0 	2 	2010 	0 	4 	85400 	85400 	97500
12 	1473 	160 	4 	69.0 	1680 	1 	1 	3 	3 	0 	... 	0 	4 	0 	3 	2010 	8 	4 	83500 	83500 	125500
13 	1474 	160 	3 	91.0 	2280 	1 	1 	3 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	148500 	148500 	148500
14 	1475 	120 	3 	69.0 	2280 	1 	1 	3 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	147500 	147500 	125500
15 	1476 	60 	3 	51.0 	12858 	1 	1 	0 	3 	0 	... 	0 	4 	0 	1 	2010 	6 	5 	147500 	147500 	345000
16 	1477 	20 	3 	69.0 	12883 	1 	1 	0 	3 	0 	... 	0 	4 	0 	6 	2010 	6 	5 	287090 	287090 	239000
17 	1478 	20 	3 	72.0 	11520 	1 	1 	3 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	287090 	287090 	337500
18 	1479 	20 	3 	66.0 	14122 	1 	1 	0 	3 	0 	... 	0 	4 	0 	2 	2010 	8 	4 	240000 	240000 	378500
19 	1480 	20 	3 	70.0 	14300 	1 	1 	3 	1 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	466500 	466500 	374000
20 	1481 	60 	3 	101.0 	13650 	1 	1 	3 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	228500 	228500 	395000
21 	1482 	120 	3 	57.0 	7132 	1 	1 	0 	3 	0 	... 	0 	4 	0 	4 	2010 	8 	4 	202500 	202500 	200141
22 	1483 	20 	3 	75.0 	18494 	1 	1 	0 	3 	0 	... 	0 	4 	0 	1 	2010 	8 	4 	174000 	174000 	221000
23 	1484 	120 	3 	44.0 	3203 	1 	1 	3 	3 	0 	... 	0 	4 	0 	1 	2010 	8 	4 	160200 	160200 	160200
24 	1485 	80 	3 	69.0 	13300 	1 	1 	0 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	184100 	184100 	160000
25 	1486 	60 	3 	110.0 	8577 	1 	1 	0 	3 	0 	... 	0 	4 	0 	4 	2010 	8 	4 	266500 	266500 	260400
26 	1487 	60 	3 	60.0 	17433 	1 	1 	1 	3 	0 	... 	0 	4 	0 	1 	2010 	8 	4 	185000 	185000 	217500
27 	1488 	20 	3 	98.0 	8987 	1 	1 	3 	3 	0 	... 	0 	4 	0 	5 	2010 	8 	4 	287090 	287090 	205000
28 	1489 	20 	1 	47.0 	9215 	1 	1 	3 	3 	0 	... 	0 	4 	0 	4 	2010 	6 	5 	208300 	208300 	208300
29 	1490 	20 	1 	60.0 	10440 	1 	1 	3 	3 	0 	... 	0 	4 	0 	5 	2010 	8 	4 	185000 	185000 	235000
30 	1491 	60 	3 	50.0 	11920 	1 	1 	3 	3 	0 	... 	0 	4 	0 	4 	2010 	8 	4 	190000 	190000 	193000
31 	1492 	30 	2 	69.0 	9800 	1 	1 	3 	3 	0 	... 	0 	4 	0 	4 	2010 	8 	4 	91000 	91000 	125500
32 	1493 	20 	3 	85.0 	15410 	1 	1 	0 	3 	0 	... 	0 	0 	0 	4 	2010 	8 	0 	175000 	175000 	160000
33 	1494 	60 	3 	70.0 	13143 	1 	1 	0 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	240000 	240000 	250000
34 	1495 	60 	3 	60.0 	11134 	1 	1 	3 	3 	0 	... 	0 	4 	0 	6 	2010 	8 	4 	240000 	240000 	293077
35 	1496 	120 	1 	108.0 	4835 	1 	1 	0 	3 	0 	... 	0 	4 	0 	3 	2010 	8 	4 	210000 	210000 	226000
36 	1497 	160 	1 	112.0 	3515 	1 	2 	3 	3 	0 	... 	0 	4 	0 	1 	2010 	8 	4 	171900 	171900 	171900
37 	1498 	160 	1 	74.0 	3215 	1 	2 	3 	3 	0 	... 	0 	4 	0 	4 	2010 	3 	4 	151000 	151000 	151000
38 	1499 	160 	1 	68.0 	2544 	1 	2 	3 	3 	0 	... 	0 	4 	0 	2 	2010 	8 	4 	151000 	151000 	151000
39 	1500 	160 	1 	65.0 	2544 	1 	2 	3 	3 	0 	... 	0 	4 	0 	5 	2010 	8 	4 	147400 	147400 	147400

40 rows × 81 columns

y = train['SalePrice']
X = train.drop('SalePrice', axis = 1)
y_pred1 = model.predict(X)
y_pred2 = forest.predict(X)
y_pred3 = tree.predict(X)
X['SalePrice_OG'] = y
X['SalePrice_model'] = y_pred2
X['SalePrice_forest'] = y_pred2
X['SalePrice_tree'] = y_pred3
X.head(40)

	Id 	MSSubClass 	MSZoning 	LotFrontage 	LotArea 	Street 	Alley 	LotShape 	LandContour 	Utilities 	... 	Fence 	MiscVal 	MoSold 	YrSold 	SaleType 	SaleCondition 	SalePrice_OG 	SalePrice_model 	SalePrice_forest 	SalePrice_tree
0 	1 	60 	3 	65.0 	8450 	1 	1 	3 	3 	0 	... 	4 	0 	2 	2008 	8 	4 	208500 	208500 	208500 	208500
1 	2 	20 	3 	80.0 	9600 	1 	1 	3 	3 	0 	... 	4 	0 	5 	2007 	8 	4 	181500 	160000 	160000 	215000
2 	3 	60 	3 	68.0 	11250 	1 	1 	0 	3 	0 	... 	4 	0 	9 	2008 	8 	4 	223500 	208500 	208500 	335000
3 	4 	70 	3 	60.0 	9550 	1 	1 	0 	3 	0 	... 	4 	0 	2 	2006 	8 	0 	140000 	140000 	140000 	140000
4 	5 	60 	3 	84.0 	14260 	1 	1 	0 	3 	0 	... 	4 	0 	12 	2008 	8 	4 	250000 	290000 	290000 	290000
5 	6 	50 	3 	85.0 	14115 	1 	1 	0 	3 	0 	... 	2 	700 	10 	2009 	8 	4 	143000 	129900 	129900 	144000
6 	7 	20 	3 	75.0 	10084 	1 	1 	3 	3 	0 	... 	4 	0 	8 	2007 	8 	4 	307000 	307000 	307000 	307000
7 	8 	60 	3 	69.0 	10382 	1 	1 	0 	3 	0 	... 	4 	350 	11 	2009 	8 	4 	200000 	200000 	200000 	200000
8 	9 	50 	4 	51.0 	6120 	1 	1 	3 	3 	0 	... 	4 	0 	4 	2008 	8 	0 	129900 	129900 	129900 	129900
9 	10 	190 	3 	50.0 	7420 	1 	1 	3 	3 	0 	... 	4 	0 	1 	2008 	8 	4 	118000 	141000 	141000 	140000
10 	11 	20 	3 	70.0 	11200 	1 	1 	3 	3 	0 	... 	4 	0 	2 	2008 	8 	4 	129500 	129500 	129500 	129500
11 	12 	60 	3 	85.0 	11924 	1 	1 	0 	3 	0 	... 	4 	0 	7 	2006 	6 	5 	345000 	345000 	345000 	345000
12 	13 	20 	3 	69.0 	12968 	1 	1 	1 	3 	0 	... 	4 	0 	9 	2008 	8 	4 	144000 	144000 	144000 	144000
13 	14 	20 	3 	91.0 	10652 	1 	1 	0 	3 	0 	... 	4 	0 	8 	2007 	6 	5 	279500 	279500 	279500 	279500
14 	15 	20 	3 	69.0 	10920 	1 	1 	0 	3 	0 	... 	1 	0 	5 	2008 	8 	4 	157000 	143000 	143000 	153500
15 	16 	45 	4 	51.0 	6120 	1 	1 	3 	3 	0 	... 	0 	0 	7 	2007 	8 	4 	132000 	132000 	132000 	132000
16 	17 	20 	3 	69.0 	11241 	1 	1 	0 	3 	0 	... 	4 	700 	3 	2010 	8 	4 	149000 	149000 	149000 	149000
17 	18 	90 	3 	72.0 	10791 	1 	1 	3 	3 	0 	... 	4 	500 	10 	2006 	8 	4 	90000 	90000 	90000 	90000
18 	19 	20 	3 	66.0 	13695 	1 	1 	3 	3 	0 	... 	4 	0 	6 	2008 	8 	4 	159000 	155000 	155000 	155000
19 	20 	20 	3 	70.0 	7560 	1 	1 	3 	3 	0 	... 	2 	0 	5 	2009 	0 	0 	139000 	127000 	127000 	115000
20 	21 	60 	3 	101.0 	14215 	1 	1 	0 	3 	0 	... 	4 	0 	11 	2006 	6 	5 	325300 	325300 	325300 	325300
21 	22 	45 	4 	57.0 	7449 	1 	0 	3 	0 	0 	... 	0 	0 	6 	2007 	8 	4 	139400 	139400 	139400 	139400
22 	23 	20 	3 	75.0 	9742 	1 	1 	3 	3 	0 	... 	4 	0 	9 	2008 	8 	4 	230000 	230000 	230000 	230000
23 	24 	120 	4 	44.0 	4224 	1 	1 	3 	3 	0 	... 	4 	0 	6 	2007 	8 	4 	129900 	129900 	129900 	129900
24 	25 	20 	3 	69.0 	8246 	1 	1 	0 	3 	0 	... 	2 	0 	5 	2010 	8 	4 	154000 	154000 	154000 	154000
25 	26 	20 	3 	110.0 	14230 	1 	1 	3 	3 	0 	... 	4 	0 	7 	2009 	8 	4 	256300 	256300 	256300 	256300
26 	27 	20 	3 	60.0 	7200 	1 	1 	3 	3 	0 	... 	4 	0 	5 	2010 	8 	4 	134800 	134800 	134800 	134800
27 	28 	20 	3 	98.0 	11478 	1 	1 	3 	3 	0 	... 	4 	0 	5 	2010 	8 	4 	306000 	297000 	297000 	372402
28 	29 	20 	3 	47.0 	16321 	1 	1 	0 	3 	0 	... 	4 	0 	12 	2006 	8 	4 	207500 	207500 	207500 	207500
29 	30 	30 	4 	60.0 	6324 	1 	1 	0 	3 	0 	... 	4 	0 	5 	2008 	8 	4 	68500 	98000 	98000 	86000
30 	31 	70 	0 	50.0 	8500 	1 	2 	3 	3 	0 	... 	2 	0 	7 	2008 	8 	4 	40000 	40000 	40000 	40000
31 	32 	20 	3 	69.0 	8544 	1 	1 	0 	3 	0 	... 	2 	0 	6 	2008 	8 	4 	149350 	110000 	110000 	159500
32 	33 	20 	3 	85.0 	11049 	1 	1 	3 	3 	0 	... 	4 	0 	1 	2008 	8 	4 	179900 	179900 	179900 	179900
33 	34 	20 	3 	70.0 	10552 	1 	1 	0 	3 	0 	... 	4 	0 	4 	2010 	8 	4 	165500 	165500 	165500 	165500
34 	35 	120 	3 	60.0 	7313 	1 	1 	3 	3 	0 	... 	4 	0 	8 	2007 	8 	4 	277500 	266000 	266000 	226000
35 	36 	60 	3 	108.0 	13418 	1 	1 	3 	3 	0 	... 	4 	0 	9 	2006 	8 	4 	309000 	309000 	309000 	309000
36 	37 	20 	3 	112.0 	10859 	1 	1 	3 	3 	0 	... 	4 	0 	6 	2009 	8 	4 	145000 	145000 	145000 	145000
37 	38 	20 	3 	74.0 	8532 	1 	1 	3 	3 	0 	... 	4 	0 	10 	2009 	8 	4 	153000 	153000 	153000 	153000
38 	39 	20 	3 	68.0 	7922 	1 	1 	3 	3 	0 	... 	4 	0 	1 	2010 	8 	0 	109000 	109000 	109000 	109000
39 	40 	90 	3 	65.0 	6040 	1 	1 	3 	3 	0 	... 	4 	0 	6 	2008 	8 	1 	82000 	100000 	100000 	106000

40 rows × 82 columns

from scipy import stats
from scipy.stats import norm, skew
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

import xgboost as xgb
# model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) 
model_xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.1, 
                             verbosity=1, silent=None, objective='reg:linear', booster='gbtree', 
                             n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                             subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.2, reg_lambda=1.2, 
                             scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain') 

model_xgb.fit(X_train, y_train)

[06:50:10] WARNING: /workspace/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
[06:50:22] WARNING: /workspace/src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=2000, n_jobs=-1, nthread=-1, num_parallel_tree=1,
             objective='reg:linear', random_state=0, reg_alpha=0.2,
             reg_lambda=1.2, scale_pos_weight=1, seed=0, silent=None,
             subsample=0.8, tree_method='exact', validate_parameters=1,
             verbosity=1)

sub = pd.DataFrame()
sub['Id'] = test1['Id']
sub['SalePrice'] = test1['SalePrice_tree']
sub.to_csv('submission.csv',index=False)
sub.head()

	Id 	SalePrice
0 	1461 	125500
1 	1462 	165500
2 	1463 	175000
3 	1464 	173000
4 	1465 	185500

 

