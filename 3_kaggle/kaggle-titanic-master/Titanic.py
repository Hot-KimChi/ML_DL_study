import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


## pandas setting change.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


## bar-chart plot 데이터를 구현하기 위해 function.
def func_bar_chart(feature):
    try:
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['Survived', 'Dead']
        df.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.tight_layout()
        plt.show()

    except():
        print('error:func_bar_chart')


## facet-grid 데이터 구현하기 위해 function
def func_facet(feature):
    try:
        facet = sns.FacetGrid(train, hue="Survived", aspect=4)
        facet.map(sns.kdeplot, feature, shade=True)
        facet.set(xlim=(0, train[feature].max()))
        facet.add_legend()

        plt.tight_layout()
        plt.show()

    except():
        print('error: func_facet')



    ## <Feature engineering: 의미있는 정보로 변경해서 모델링을 진행한다.>
    ## Feature engineering is the process of using domain knowledge of the data to create features (feature vectors) that make machine learning algorithms work.
    ## 1) <Pclass는 key feature for classifier. 침수가 3 class 바닥부터 시작되었기 때문에. ==> Pclass는 꼭 사용 !!!
    ## 2) Name에서 결혼한 여성일 경우, 생존 확률이 높을 것이다. 아래 구현.
def func_feature_eng(train, test):
    try:
        # 2번 작업하는 번거스러움을 피하기 위해, 합쳐서 1번만 작업 진행.
        train_test_data = [train, test]

        ## Name을 기반으로 결혼한 여성일 경우, "Mr": 0, "Miss": 1, "Mrs": 2 표시. 그 외 3으로 define.
        for dataset in train_test_data:
            dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        print('Title 추가:')
        print(dataset.head())

        ## train_test_data에서 dataset으로 생성할지라도, train / test의 컬럼(Title)이 생성된다.
        print(train.head())
        print(train['Title'].value_counts())
        print(test['Title'].value_counts())

        ## Title mapping.
        title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                         "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                         "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}

        for dataset in train_test_data:
            dataset['Title'] = dataset['Title'].map(title_mapping)
        print('Title mapping:')
        print(train.head())

        # Delete unnecessary feature from dataset
        train.drop('Name', axis=1, inplace=True)
        test.drop('Name', axis=1, inplace=True)


        ## Sex를 male: 0 female: 1 으로 변환.
        sex_mapping = {"male": 0, "female": 1}
        for dataset in train_test_data:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        print('Sex mapping:')
        print(train.head())


        ## Age(나이), 어떤 데이터는 missing.
        ## missing 데이터를 어떠한 방법으로 채워 넣을 것인가? 1) 전체 나이의 평균으로 채워넣기 2) 남자의 평균 / 결혼여성 평균 / 결혼하지 않는 여성의 평균.
        ## 2번 method로 진행.
        train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
        test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)


        ## Binning(라벨 / 원핫 인코딩을 쓰지 않고 변환)
        for dataset in train_test_data:
            dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,



    except():
        print('error: func_feature_eng')


if __name__ == '__main__':

    ## Part1: Exploratory Data Analysis(EDA)
    ## 1)Analysis of the features.
    ## 2)Finding any relations or trends considering multiple features.

    ## load data
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    ## data parameter 설명.
    print(train.head(5))
    print(train.info())

    print(test.head(5))
    print(test.info())
    print()

    ## Survived: 0 = No, 1 = Yes
    ## pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
    ## sibsp: # of siblings / spouses aboard the Titanic(와이프나 사촌)
    ## parch: # of parents / children aboard the Titanic(아이 혹은 부모)
    ## ticket: Ticket number
    ## cabin: Cabin number
    ## embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

    print(train.shape)
    print(test.shape)

    ## preprocessing: null 데이터에 대해서 어떻게 handling할 건지 고민.
    print(train.isnull().sum())
    print(test.isnull().sum())


    ## bar_chart for categorical features
    func_bar_chart('Sex')
    func_bar_chart('Pclass')
    func_bar_chart('SibSp')
    func_bar_chart('Parch')

    ## feature engineering.
    func_feature_eng(train, test)

    ## Title을 bar_chart에 plot.
    func_bar_chart('Title')
    func_facet('Age')

    ##