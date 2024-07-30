import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
import statsmodels.formula.api as smf
from sklearn import tree
from scipy.stats import tukey_hsd
from scipy import stats
import seaborn as sns

def main():
    print("Data manipulation starts..")
    df = pd.read_excel(r"C:\Users\domin\Documents\Bachelorarbeit\data_for_python.xlsx", index_col=None)
    original_table_shape = df.shape
    df_description = df.iloc[0]

    df = privacy_data_cleaning(df)

    df = data_cleaning(df)
    print("Cleaned table: ", df.shape)
    df = recode_inverted_variables(df)

    
    #add needed variables
    new_variables = ['CarInstrumentalPerception',
                     'CarFinancialRisk',
                     'CarSocialValue',
                     'CarEmotionalInvestement',
                     'EVAttitude',
                     'Materialism',
                     'DeownershipOrientation',
                     'PsychologicalOwnership',
                     'Innovativeness',
                     'TechnologyPhobia',
                     'TechnologyInterest',
                     'FODAttitude',
                     'FODFinancialAppeal',
                     'FunctionImportance',
                     'FunctionAcceptance',
                     'PackageImportance',
                     'PackageAcceptance']
    items = [['CA10_08','CA10_09','CA10_10','CA10_11'],
             ['CA10_12','CA10_13','CA10_14','CA10_15'],
             ['CA07_01','CA07_02','CA07_03','CA07_04'],
             ['CA07_05','CA07_06','CA07_07'],
             ['CA07_18','CA07_19','CA07_20'],
             ['CO02_04','CO02_05','CO02_06','CO02_07','CO02_08','CO02_09'],
             ['CO02_10','CO02_11','CO02_12'],
             ['CS10_01','CS10_02'],
             ['CT01_01','CT01_02','CT01_03'],
             ['CT01_04','CT01_05','CT01_06'],
             ['CT01_07','CT01_08','CT01_09'],
             ['FA01_03','FA01_04','FA01_05','FA01_06'],
             ['FA01_07','FA01_08','FA01_09','FA01_10'],
             ['FF01_01','FF01_02','FF01_03','FF01_04','FF01_05','FF01_06','FF01_07','FF01_08','FF01_09','FF01_10','FF01_11','FF01_12','FF01_13','FF01_14','FF01_15'],
             ['FF03_01','FF03_02','FF03_03','FF03_04','FF03_05','FF03_06','FF03_07','FF03_08','FF03_09','FF03_10','FF03_11','FF03_12','FF03_13','FF03_14','FF03_15'],
             ['FP01_01','FP01_02','FP01_03','FP01_04','FP01_05','FP01_06'],
             ['FP03_01','FP03_02','FP03_03','FP03_04','FP03_05','FP03_06']]
    
    # demographic variables, 'Age' is calculated elsewhere
    demographics_continuous = {'CA03_01': 'DailyDrivingKM',
                               'CA03_02': 'WeeklyDrivingKM',
                               'CD17': 'Income',
                               'CD24': 'CitySize',
                               'CD25': 'CityDistance',
                               'FA02_01': 'CarSubscriptionExp',
                               'FA02_02': 'FODExp',
                               'CA05_02': 'HouseholdCars',
                               'CA05_01': 'HouseholdDrivers',
                               'CD27_01': 'SES'}
    demographics_categorical = {'CA01': 'DrivingLicense',
                                'CA04': 'PurchaseExp',
                                'CA06': 'CarType',
                                'CD01': 'Gender',
                                'CD26': 'Education',
                                'CA08_01': 'RideTypeCommute',
                                'CA08_02': 'RideTypeShopping',
                                'CA08_03': 'RideTypeChild',
                                'CA08_04': 'RideTypeLeisure',
                                'CA09_01': 'PurchasePrioPrice',
                                'CA09_02': 'PurchasePrioQuality',
                                'CA09_03': 'PurchasePrioBrand',
                                'CA09_04': 'PurchasePrioFunctions',
                                'CA09_05': 'PurchasePrioGuarantee',
                                'CD08': 'Country',
                                'CD19': 'BirthPlace',
                                'CD14': 'Occupation',
                                'CD20_01': 'Culture',
                                'CD23':'Accommodation'}
    df.rename(columns=demographics_continuous, inplace=True)
    df.rename(columns=demographics_categorical, inplace=True)
    


    df = add_variables(df, new_variables, items)
    print("New variables table: ", df.shape)
    df_variable_description = describe_variables(df, new_variables, items)
    if 0:
        print(df_variable_description)
        print(df_variable_description.to_latex(index=False,
                                            formatters={"name":str.upper},
                                            float_format="{:.2f}".format))
        print(df_variable_description.sort_values('alpha'))


    # general attitudes

    
    # new_var_info = pd.DataFrame({'name': new_variables,'mean': mean,})

    # data analysis
    # to-do: correlation, anova (Trzebinski), process (Trzebinski), decision tree, regression, cluster analysis?, group identificaiton

    # political leaning
    demographic_variables = ['CA01','CA12','CA02','CA03_01','CA03_02','CA04','CA06','CA08_01','CA08_02','CA08_03','CA08_04','CA09_01','CA09_02','CA09_03','CA09_04','CA09_05','CA05_01','CA05_02','CA05_03','CD01','CD04_01','CD08','CD08s','CD19','CD19s','CD14','CD14_08','CD17','CD18_01','CD20_01','CD23','CD23_10','CD24','CD25','CD26','CD26_08','CD27_01','CD30_pts','CD30_rgs','CD30_01']
    df[['Right', 'Libertarian','political_points']] = df['CD30_pts'].str.split(',',expand=True)
    df['Right'] = df['Right'].astype(float)/440
    df['Libertarian'] = df['Libertarian'].astype(float)/480
    print("Manipulated table: ", df.shape)
    print("Originally imported table: ", original_table_shape)
    demographic_variables.append('Right')
    demographic_variables.append('Libertarian')
    # age
    pd.to_numeric(df['CD04_01'], errors='coerce')
    for index, row in df.iterrows():
        tmp = df.loc[index, 'CD04_01']
        tmp = float(tmp)
        #print("tmp: ", tmp)
        if tmp > 1900 and tmp < 2007:
            df.at[index, 'Age'] = 2024 - tmp
        else:
            df.at[index, 'Age'] = None
    # cars per driver
    df['CarsPerDriver'] = df['HouseholdCars'].astype(float)/df['HouseholdDrivers'].astype(float)


    

    # RQ2: DEMOGRAPHIC ANALYSIS
    if 0:
        print_mean_std_perc(df,['Age', 'Gender', 'CarSubscriptionExp','FODExp','DrivingLicense','PurchaseExp','CitySize','CityDistance'])

    # correlation analysis
    alphas = [0.1,0.05,0.01]
    for alpha in alphas:
        t_value=stats.t.ppf(1-(alpha/2),len(df.index))
        min_significant_correlation = (1/(((len(df.index)-2)/t_value**2)-1))**(1/2)
        print(alpha, " t_val: ", t_value, " sig corr: ", min_significant_correlation)

    demographic_correlation_list = ['Age', 'Income', 'SES', 'Right', 'Libertarian', 'CitySize', 'CityDistance']
    car_usage_correlation_list = ['DailyDrivingKM', 'WeeklyDrivingKM', 'CarsPerDriver', 'HouseholdCars', 'HouseholdDrivers', 'CarSubscriptionExp', 'FODExp']
    correlation_df = df[demographic_correlation_list+['FunctionAcceptance', 'PackageAcceptance']]
    #print(correlation_df.corr())
    #plt.matshow(correlation_df.corr())
    #plt.show()
    correlation_df = df[car_usage_correlation_list+['FunctionAcceptance', 'PackageAcceptance']]
    #print(correlation_df.corr())
    #plt.matshow(correlation_df.corr())
    #plt.show()
    if 0:
        print("['CarSubscriptionExp', 'FunctionAcceptance']")
        df_modified = df[['CarSubscriptionExp', 'FunctionAcceptance']].dropna(how='any') #how='all' also possible
        print(stats.pearsonr(df_modified['CarSubscriptionExp'], df_modified['FunctionAcceptance']))
        print("['CarSubscriptionExp', 'PackageAcceptance']")
        df_modified = df[['CarSubscriptionExp', 'PackageAcceptance']].dropna(how='any') #how='all' also possible
        print(stats.pearsonr(df_modified['CarSubscriptionExp'], df_modified['PackageAcceptance']))
        print("['Libertarian', 'FunctionAcceptance']")
        df_modified = df[['Libertarian', 'FunctionAcceptance']].dropna(how='any') #how='all' also possible
        print(stats.pearsonr(df_modified['Libertarian'], df_modified['FunctionAcceptance']))
        print("['SES', 'PackageAcceptance']")
        df_modified = df[['SES', 'PackageAcceptance']].dropna(how='any') #how='all' also possible
        print(stats.pearsonr(df_modified['SES'], df_modified['PackageAcceptance']))

    if 0:
        plt.plot('Age', 'FunctionAcceptance', data=df, marker='o', linewidth=0)
        plt.title("FOD Acceptance by Age")
        plt.xlabel("Age")
        plt.ylabel("Acceptance of functions-on-demand")
        plt.show()
    
    # ANOVA analysis https://www.reneshbedre.com/blog/anova.html?utm_content=cmp-true
    categorical_variables = list(demographics_categorical.values())
    if 0:
        ax = sns.boxplot(x='CarType', y='FunctionAcceptance', data=df)
        ax = sns.swarmplot(x='CarType', y='FunctionAcceptance', data=df)
        plt.show()

    dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
    for variable in dependent_variables:
        df[variable]=pd.to_numeric(df[variable], errors='coerce')
    
    df['CarType']=pd.to_numeric(df['CarType'], errors='coerce')
    for independent_variable in categorical_variables:
        for dependent_variable in dependent_variables:
            relationship = independent_variable + ' ~ ' + dependent_variable
            print(relationship)
            model = ols('CarType ~ FunctionAcceptance', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)

    # Kruskal-Wallis Test - non-parametric equivalent of one-way ANOVA
    df_modified = df[['CarType', 'FunctionAcceptance']]
    #df_modified = df_modified.groupby('CarType')['FunctionAcceptance'].apply(list).reset_index()
    df_modified = df_modified.pivot(columns='CarType', values='FunctionAcceptance')
    groups = []
    for column_name in df_modified.columns.values:
        groups.append(df_modified[column_name].dropna(how='any').to_list())
    stat, p_value = stats.kruskal(groups[0],groups[1],groups[2])
    print('Kruskal-Wallis: stat: ', stat, ' p_value: ', p_value)

    # Mann-Whitney U test - returns NAN 
    print('Mann-Whitney: ', stats.mannwhitneyu(x=df['Age'], y=df['FunctionAcceptance'], alternative = 'two-sided')) #other alternative: greater
    
    # Levene Test
    df_modified = df[['CarType', 'FunctionAcceptance']]
    #df_modified = df_modified.groupby('CarType')['FunctionAcceptance'].apply(list).reset_index()
    df_modified = df_modified.pivot(columns='CarType', values='FunctionAcceptance')
    groups = []
    for column_name in df_modified.columns.values:
        groups.append(df_modified[column_name].dropna(how='any').to_list())
    stat, p_value = stats.levene(groups[0],groups[1],groups[2])
    print('Levene: stat: ', stat, ' p_value: ', p_value)

    # MANOVA https://www.reneshbedre.com/blog/manova-python.html
    model = MANOVA.from_formula('CarType + Education ~ FunctionAcceptance', data=df)
    print(model.mv_test()) # Pillai's trace is relevant

    # POST HOC tests for significant ANOVA
    #print(model.summary())
    df_modified = df[['CarType', 'FunctionAcceptance']]
    data = pd.get_dummies(df_modified, columns=['CarType'], drop_first=False, dtype=float)
    X = data.drop(columns=['FunctionAcceptance'])
    y = data['FunctionAcceptance']
    X = sm.add_constant(X)
    model = sm.OLS(y,X).fit()
    #print(model.summary())

    # POST HOX test for significant MANOVA


    # Decision Tree 

    X, y = df[['Gender']],df['FunctionAcceptance']
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X,y)
    plt.figure()
    tree.plot_tree(clf)
    #plt.show()

    # regressions - mediation analysis (pingouin, statsmodels), moderation analysis (statsmodels, process), 



def privacy_data_cleaning(df):
    # information about the study subject hours
    df.drop('EN01_01', axis=1, inplace=True)
    df.drop('VP02_01', axis=1, inplace=True)
    df.drop('VP02_02', axis=1, inplace=True)
    df.drop('VP02_03', axis=1, inplace=True)
      
    return df

def data_cleaning(df):
    #drop label row
    df = df.drop(0)
    #drop uncomplete surveys
    df = df.drop(df[df.FINISHED==0].index)
    #drop record by author (CASE=43)
    df = df.drop(df[df.CASE==43].index)
    #drop wrongly answered check_questions
    df = df.drop(df[df.FA01_14!=4].index)
    return df

def recode_inverted_variables(df):
    #recode variables
    recode_list = ["CA07_18", "CT01_02", "CT01_03", "FA01_10"]
    for variable in recode_list:
        df[variable] = 7-df[variable]
    return df

def add_variables(df, new_variables, items):
    i=0
    for variable in new_variables:
        item = items[i]
        df[variable] = df[item].mean(axis=1)
        i+=1
    return df

def describe_variables(df, new_variables, items):
    #Cronbach's Alpha https://statisticalpoint.com/cronbachs-alpha-in-python/
    alphas = []
    for item in items:
        data = df[item]
        data = data.dropna(how='any') #how='any' also possible
        #print(data)
        #alpha = (len(item)/(len(item)-1))*(1-(()/))
        alpha_whole=pg.cronbach_alpha(data=data.fillna(data.median(numeric_only=True)).infer_objects())
        alpha=alpha_whole[0]
        alpha = round(alpha,3)
        alphas.append(alpha)

    #mean of new variables
    mean = []
    for var in new_variables:
        mean.append(df.loc[:,var].mean())

    #number of items
    item_length = []
    for item in items:
        item_length.append(len(item))

    new_var_info = pd.DataFrame(
        {'name': new_variables,
         'alpha': alphas,
         'mean': mean,
         'items': item_length

         #'items':items
         }
    )
    return new_var_info

def find_nth(haystack: str, needle: str, n: int) -> int: #https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def print_mean_std_perc(df, columns):
    for column in columns:
        print(column)
        print("usage of subscription: ", df[column].mean(), " std: ", df[column].std())
        print("percentage: ", (df[column].value_counts()/len(df.index)))

if __name__ == '__main__':
    main()