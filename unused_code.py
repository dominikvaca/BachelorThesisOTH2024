

# RQ 5
# https://www.statsmodels.org/stable/generated/statsmodels.stats.mediation.Mediation.html
# mediation analysis with statsmodels
# works but too long
independent_variable = 'Importance'
mediation_variable = 'Tangibility'
dependent_variable = 'Acceptance'
Probit = links.Probit
outcome_formula = dependent_variable + ' ~ ' + mediation_variable + ' + ' + independent_variable
mediator_formula = mediation_variable + ' ~ ' + independent_variable
outcome_model = sm.GLM.from_formula(outcome_formula, df_tangibility, family=sm.families.Binomial(link=Probit()))
mediator_model = sm.OLS.from_formula(mediator_formula, data=df_tangibility)
med = Mediation(outcome_model, mediator_model, independent_variable, mediation_variable).fit()
print(med.summary())


# Mann-Whitney U test - returns NAN 
    # print('Mann-Whitney: ', stats.mannwhitneyu(x=df['Age'], y=df['FunctionAcceptance'], alternative = 'two-sided'))


# regression with categorical independent variable
#not used, not working
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