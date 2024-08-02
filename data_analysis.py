import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.mediation import Mediation
import statsmodels.formula.api as smf
import statsmodels.genmod.families.links as links

from sklearn import tree
from scipy.stats import f_oneway, tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import seaborn as sns
from pingouin import mediation_analysis

def main():
    print("Data manipulation starts..")
    df = pd.read_excel(r"C:\Users\domin\Documents\Bachelorarbeit\data_survey.xlsx", index_col=None)
    original_table_shape = df.shape
    df_description = df.iloc[0]

    df = privacy_data_cleaning(df)

    df = data_cleaning(df)
    print("Cleaned table: ", df.shape)
    df = recode_inverted_variables(df)
    df = recode_driving_license(df)

    
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
    

    #categorical variables
    independent_variables = list(demographics_categorical.values())
    independent_variables.remove('Culture')
    dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
    # ANOVA analysis https://www.reneshbedre.com/blog/anova.html?utm_content=cmp-true
    #plot_variance(df, 'CarType', 'FunctionAcceptance')
    if 0:
        anova_printout(df, independent_variables, dependent_variables)
        # 'CarType ~ PackageAcceptance' 0.068238
        # 'RideTypeShopping ~ FunctionAcceptance' 0.074672
        # PurchasePrioFunctions ~ PackageAcceptance 0.062608
            

    # Kruskal-Wallis Test - non-parametric equivalent of one-way ANOVA
    #print(independent_variables)
    RideTypeNames = ['RideTypeCommute', 'RideTypeShopping', 'RideTypeChild', 'RideTypeLeisure']
    PurchasePrioNames = ['PurchasePrioGuarantee', 'PurchasePrioFunctions', 'PurchasePrioBrand', 'PurchasePrioQuality', 'PurchasePrioPrice']
    dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
    if 0:
        deleted_variables = RideTypeNames + PurchasePrioNames
        for var in deleted_variables:
            independent_variables.remove(var)
    if 0:
        print("Kruskal-Wallis Test")
        df_kruskal = kruskal_get_results(df, independent_variables, dependent_variables)
        print(df_kruskal)
    if 0: #print a LaTeX type table 
        print(df_kruskal.to_latex(index=False,
                                formatters={"name":str.upper},
                                float_format="{:.2f}".format))

    # Mann-Whitney U test - returns NAN 
    # print('Mann-Whitney: ', stats.mannwhitneyu(x=df['Age'], y=df['FunctionAcceptance'], alternative = 'two-sided')) #other alternative: greater
    
    # Levene Test
    if 0:
        df_modified = df[['CarType', 'FunctionAcceptance']]
        #df_modified = df_modified.groupby('CarType')['FunctionAcceptance'].apply(list).reset_index()
        df_modified = df_modified.pivot(columns='CarType', values='FunctionAcceptance')
        groups = []
        for column_name in df_modified.columns.values:
            groups.append(df_modified[column_name].dropna(how='any').to_list())
        stat, p_value = stats.levene(groups[0],groups[1],groups[2])
        print('Levene: stat: ', stat, ' p_value: ', p_value)

    # MANOVA https://www.reneshbedre.com/blog/manova-python.html
    if 0:
        model = MANOVA.from_formula('CarType + Education ~ FunctionAcceptance', data=df)
        print(model.mv_test()) # Pillai's trace is relevant
        print(model.summary())

    # TUKEY - POST HOC tests for significant ANOVA
    if 0:
        significant_independent_variables = df_kruskal.loc[df_kruskal['p_value'] < 0.1,'independent_var'].to_list()
        # inspo: df.loc[((df[geo_variable] != 276) & (df[geo_variable] != 203)), geo_variable]=3 # code for other countries
        dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
        print("post hoc for anova")
        print("indep: ", significant_independent_variables, "\ndep: ", dependent_variables)
        tukey_printout(df, significant_independent_variables, dependent_variables)
        # interesting relationships
        
        if 0: #not used, not working
            df_modified = df[['CarType', 'FunctionAcceptance']]
            data = pd.get_dummies(df_modified, columns=['CarType'], drop_first=False, dtype=float)
            #print(data)
            X = data.drop(columns=['FunctionAcceptance'])
            y = data['FunctionAcceptance']
            X = sm.add_constant(X)
            tukey = pairwise_tukeyhsd(endog=y, groups=X, alpha=0.05)
            print(tukey)
            model = sm.OLS(y,X).fit()
            print(model.summary())

    # RQ3: What psychological concepts influence the functions-on-demand acceptance?
    if 0:
        dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
        independent_variables = new_variables
        
        for variable in independent_variables + dependent_variables:
            print(variable, ", mean: ", df[variable].mean(), ", std: ", df[variable].std())
        iteration = 0
        for independent_variable in independent_variables:
            for dependent_variable in dependent_variables:
                if independent_variable != dependent_variable:
                    print(iteration, ". ",independent_variable, " ~ ", dependent_variable)
                    df_modified = df[[independent_variable, dependent_variable]].dropna(how='any') #how='all' also possible
                    corr, p_value = stats.pearsonr(df_modified[independent_variable], df_modified[dependent_variable])
                    print(f'r={corr:.3f}, \\textalpha={p_value:.4f}') #format used in the paper
                    iteration += 1


    # RQ4: Difference between German and Czech Participants
    # differences of the groups
    if 0:
        geo_variables = ['Country', 'BirthPlace']
        dependent_variables = ['FunctionAcceptance', 'PackageAcceptance']
        for geo_variable in geo_variables:
            df.loc[((df[geo_variable] != 276) & (df[geo_variable] != 203)), geo_variable]=3 # code for other countries
            df.loc[(df[geo_variable] == 276), geo_variable]=1 # Germany code
            df.loc[(df[geo_variable] == 203), geo_variable]=2 # Czech Republic code
        print("STD for countries")
        for geo_variable in geo_variables:
            for dependent_variable in dependent_variables:
                print(geo_variable, "~", dependent_variable)
                for i in range(3):
                    print(i+1, "std:", df.loc[(df[geo_variable] == i+1), dependent_variable].std())
        print("Levene test for countries") # more robust test than F-test
        levene_printout(df, dependent_variables, geo_variables)
        plot_variance(df, 'Country', 'FunctionAcceptance')
        print("ANOVA for countries")
        anova_printout(df, geo_variables, dependent_variables)
        print("Kruskal for countries")
        print(kruskal_get_results(df, geo_variables, dependent_variables))
        print("Tukey for countries")
        tukey_printout(df, geo_variables, dependent_variables)
        if 0:
            plot_variance(df, geo_variables[1], dependent_variables[0])
            plot_variance(df, geo_variables[1], dependent_variables[1])

    # RQ5: role of tangibility (and importance)
    df_tangibility = pd.read_excel(r"C:\Users\domin\Documents\Bachelorarbeit\data_tangibility.xlsx", index_col=None)
    importance_functions = {'FF01_01': 'ImpFunHUD',
                            'FF01_02': 'ImpFunWIFI',
                            'FF01_03': 'ImpFunSpeech',
                            'FF01_04': 'ImpFunHeatedSeats',
                            'FF01_05': 'ImpFunDigiRadio',
                            'FF01_06': 'ImpFunPhoneApp',
                            'FF01_07': 'ImpFunAutoBeams',
                            'FF01_08': 'ImpFunMapUpdate',
                            'FF01_09': 'ImpFunAutopilot',
                            'FF01_10': 'ImpFunTrafficSign',
                            'FF01_11': 'ImpFunMotorPower',
                            'FF01_12': 'ImpFunPredMaintenance',
                            'FF01_13': 'ImpFunLog',
                            'FF01_14': 'ImpFunFuelPrices',
                            'FF01_15': 'ImpFunToll'}
    acceptance_functions = {'FF03_01': 'AccFunHUD',	
                            'FF03_02': 'AccFunWIFI',	
                            'FF03_03': 'AccFunSpeech',	
                            'FF03_04': 'AccFunHeatedSeats',	
                            'FF03_05': 'AccFunDigiRadio',	
                            'FF03_06': 'AccFunPhoneApp',	
                            'FF03_07': 'AccFunAutoBeams',	
                            'FF03_08': 'AccFunMapUpdate',	
                            'FF03_09': 'AccFunAutopilot',	
                            'FF03_10': 'AccFunTrafficSign',	
                            'FF03_11': 'AccFunMotorPower',	
                            'FF03_12': 'AccFunPredMaintenance',	
                            'FF03_13': 'AccFunLog',	
                            'FF03_14': 'AccFunFuelPrices',	
                            'FF03_15': 'AccFunToll'}
    importance_packages = {'FP01_01': 'ImpPacInfotainment',
                            'FP01_02': 'ImpPacADAS',
                            'FP01_03': 'ImpPacNavigation',
                            'FP01_04': 'ImpPacLighting',
                            'FP01_05': 'ImpPacRemote',
                            'FP01_06': 'ImpPacPerformance'}
    acceptance_packages = {'FP03_01': 'AccPacInfotainment',
                            'FP03_02': 'AccPacADAS',
                            'FP03_03': 'AccPacNavigation',
                            'FP03_04': 'AccPacLighting',
                            'FP03_05': 'AccPacRemote',
                            'FP03_06': 'AccPacPerformance'}
    imp_acc_column_names = [importance_functions, acceptance_functions, importance_packages, acceptance_packages]
    for column_name in imp_acc_column_names:
        df.rename(columns=column_name, inplace=True)

    importance_variables = list(importance_functions.values())+list(importance_packages.values())
    acceptance_variables = list(acceptance_functions.values())+list(acceptance_packages.values())
    
    # importance -> acceptance
    df_correlation = corr_get_results(df, importance_variables, acceptance_variables)
    print(df_correlation.sort_values('corr'))

    # tangibility -> acceptance
    #print(df_tangibility)
    tangibility_variables = df_tangibility['ID'].to_list()
    tangibility_values = df_tangibility['Tangibility'].to_list()
    for column_name, value in zip(tangibility_variables, tangibility_values):
        print(column_name, value)
        df[column_name]=value
        
    acceptance_means = []
    for var in acceptance_variables:
        mean = df[var].mean()
        acceptance_means.append(mean)
    df_tangibility['Acceptance'] = acceptance_means
    importance_means = []
    for var in importance_variables:
        mean = df[var].mean()
        importance_means.append(mean)
    df_tangibility['Importance'] = importance_means
    print(df_tangibility)
    corr, p_value = stats.pearsonr(df_tangibility['Tangibility'], df_tangibility['Acceptance'])
    print("corr: ", corr, "p-value: ", p_value)
    corr, p_value = stats.pearsonr(df_tangibility['RealTime'], df_tangibility['Acceptance'])
    print("corr: ", corr, "p-value: ", p_value)
    corr, p_value = stats.pearsonr(df_tangibility['Familiarity'], df_tangibility['Acceptance'])
    print("corr: ", corr, "p-value: ", p_value)

    
    # regressions - mediation analysis (pingouin, statsmodels), moderation analysis (statsmodels, process),
    # https://pingouin-stats.org/build/html/generated/pingouin.mediation_analysis.html
    # mediation analysis with pingouin
    independent_variable = 'Importance'
    mediation_variable = 'Tangibility'
    dependent_variable = 'Acceptance'
    
    print(independent_variable, "->", mediation_variable, "->", dependent_variable)
    statistic_mediation = mediation_analysis(data=df_tangibility, x=independent_variable, m=mediation_variable, y=dependent_variable, alpha=0.05) # significant if the confidence intervals do not include zero
    print(statistic_mediation)

    # https://www.statsmodels.org/stable/generated/statsmodels.stats.mediation.Mediation.html
    # mediation analysis with statsmodels
    independent_variable = 'FunctionImportance'
    mediation_variable = 'Age'
    dependent_variable = 'FunctionAcceptance'
    df_modified = df[[independent_variable, mediation_variable, dependent_variable]].dropna(how='any') #how='all' also possible

    independent_variable = 'Importance'
    mediation_variable = 'Tangibility'
    dependent_variable = 'Acceptance'
    
    print(df_modified)
    Probit = links.Probit
    outcome_formula = dependent_variable + ' ~ ' + mediation_variable + ' + ' + independent_variable
    mediator_formula = mediation_variable + ' ~ ' + independent_variable
    outcome_model = sm.GLM.from_formula(outcome_formula, df_tangibility, family=sm.families.Binomial(link=Probit()))
    mediator_model = sm.OLS.from_formula(mediator_formula, data=df_tangibility)
    med = Mediation(outcome_model, mediator_model, independent_variable, mediation_variable).fit()
    print(med.summary())

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

def recode_driving_license(df):
    #recode variables
    recode_list = ["CA01"]
    dictionary = {1: 1,
                  2: 3,
                  3: 4,
                  4: 5,
                  5: 6,
                  6: 2}
    for variable in recode_list:
        # df[variable] = df.replace({variable: dictionary})
        df[variable] = df[variable].map(dictionary)
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

def plot_variance(df, independent_variable, dependent_variable):
    ax = sns.boxplot(x=independent_variable, y=dependent_variable, data=df)
    ax = sns.swarmplot(x=independent_variable, y=dependent_variable, data=df)
    plt.show()

def anova_printout(df, independent_variables, dependent_variables):
    for variable in dependent_variables:
        df[variable]=pd.to_numeric(df[variable], errors='coerce')
    for independent_variable in independent_variables:
        df[independent_variable]=pd.to_numeric(df[independent_variable], errors='coerce')
        for dependent_variable in dependent_variables:
            relationship = independent_variable + " ~ " + dependent_variable
            print("'"+relationship+"'")
            df_modified = df[[independent_variable, dependent_variable]]
            df_modified = df_modified.dropna(how='any') #how='all' also possible
            model = ols(formula=relationship, data=df_modified).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)

def kruskal_get_results(df, independent_variables, dependent_variables):
    indep_kruskal = []
    dep_kruskal = []
    statistics_kruskal = []
    p_value_kruskal = []
    for independent_variable in independent_variables:
            #print("percentage: ", (df[variable].value_counts()/len(df.index)))
        var_num = len(df[independent_variable].value_counts())
            #print(independent_variable + ": " + str(var_num))
            
        for dependent_variable in dependent_variables:
            df_modified = df[[independent_variable, dependent_variable]]
            df_modified = df_modified.pivot(columns=independent_variable, values=dependent_variable)
            groups = []
            for column_name in df_modified.columns.values:
                groups.append(df_modified[column_name].dropna(how='any').to_list())
            if var_num == 2:
                stat, p_value = stats.kruskal(groups[0],groups[1])
            elif var_num == 3:
                stat, p_value = stats.kruskal(groups[0],groups[1],groups[2])
            elif var_num == 4:
                stat, p_value = stats.kruskal(groups[0],groups[1],groups[2],groups[3])
            elif var_num == 5:
                stat, p_value = stats.kruskal(groups[0],groups[1],groups[2],groups[3],groups[4])
            elif var_num == 6:
                stat, p_value = stats.kruskal(groups[0],groups[1],groups[2],groups[3],groups[4],groups[5])
            else:
                    # print("This variable couldn't be printed: " + independent_variable)
                continue
                #print(' -',dependent_variable, ' stat: ', stat, ' p_value: ', p_value)
            indep_kruskal.append(independent_variable)
            dep_kruskal.append(dependent_variable)
            statistics_kruskal.append(stat)
            p_value_kruskal.append(p_value)
    df_kruskal = pd.DataFrame({'independent_var':indep_kruskal,
                                    'dependent_var':dep_kruskal,
                                    'stat':statistics_kruskal,
                                    'p_value':p_value_kruskal})
    return df_kruskal

def tukey_printout(df, independent_variables, dependent_variables):
    iteration = 0
    for independent_variable in independent_variables: 
        for dependent_variable in dependent_variables:
            df_modified = df[[independent_variable, dependent_variable]]
            df_modified = df_modified.pivot(columns=independent_variable, values=dependent_variable)
            groups = []
            for column_name in df_modified.columns.values:
                group = (df_modified[column_name].dropna(how='any').to_list())
                if len(group) > 1:
                    groups.append(group)
            var_num = len(groups)
            print(" - ",iteration,". ", independent_variable, " ~ ",dependent_variable)
            if var_num == 2:
                result = tukey_hsd(groups[0],groups[1])
            elif var_num == 3:
                result = tukey_hsd(groups[0],groups[1],groups[2])
            elif var_num == 4:
                result = tukey_hsd(groups[0],groups[1],groups[2],groups[3])
            elif var_num == 5:
                result = tukey_hsd(groups[0],groups[1],groups[2],groups[3],groups[4])
            elif var_num == 6:
                result = tukey_hsd(groups[0],groups[1],groups[2],groups[3],groups[4],groups[5])
            else:
                    # print("This variable couldn't be printed: " + independent_variable)
                continue
            print(result)
                #plot_variance(df, independent_variable, dependent_variable)
            iteration +=1

def levene_printout(df, dependent_variables, geo_variables):
    for geo_variable in geo_variables:
        for dependent_variable in dependent_variables:
            print(geo_variable, "~", dependent_variable)
            df_modified = df[[geo_variable, dependent_variable]]
            #df_modified = df_modified.groupby('CarType')['FunctionAcceptance'].apply(list).reset_index()
            df_modified = df_modified.pivot(columns=geo_variable, values=dependent_variable)
            groups = []
            for column_name in df_modified.columns.values:
                groups.append(df_modified[column_name].dropna(how='any').to_list())
            stat, p_value = stats.levene(groups[0],groups[1],groups[2])
            print('Levene: stat: ', stat, ' p_value: ', p_value)

def corr_get_results(df, variables_1, variables_2):
    correlations = []
    p_values = []
    for var1, var2 in zip(variables_1, variables_2):
        df_modified = df[[var1, var2]].dropna(how='any') #how='all' also possible
        corr, p_value = stats.pearsonr(df_modified[var1], df_modified[var2])
        correlations.append(corr)
        p_values.append(p_value)
        #print(var1, " ~ ", var2)
        #print(f'r={corr:.3f}, \\textalpha={p_value:.4f}') #format used in the paper
    df_correlation = pd.DataFrame({'indep': variables_1,
                                    'dep': variables_2,
                                    'corr': correlations,
                                    'p_value': p_values})  
    return df_correlation

if __name__ == '__main__':
    main()