# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Replication code for regional estimates of HCV prevalence
# 
# From Mohd Hanafiah K, Groeger J, Flaxman AD, Wiersma ST. Global epidemiology of hepatitis C virus infection: New estimates of age-specific antibody to HCV seroprevalence. Hepatology. 2013 Apr;57(4):1333-42. doi: 10.1002/hep.26141. Epub 2013 Feb 4.
# 
# http://www.ncbi.nlm.nih.gov/pubmed/23172780

# <codecell>

# This Python code will reproduce predictions 
# for the following region/sex/year:
predict_region = 'north_africa_middle_east'
predict_sex = 'male'
predict_year = 2005

# <codecell>

# import dismod code
%cd ~/gbd_dev/gbd
import dismod3

# <codecell>

# load HCV model data
model_path = '/home/j/Project/dismod/hcv_replication/'
dm1 = dismod3.data.load(model_path)

# <codecell>

# Fit stage-1 model with heterogeneity prior set to "Very"
dm1.parameters['p']['heterogeneity'] = 'Very'
dm1.vars += dismod3.ism.age_specific_rate(dm1, 'p')
%time dismod3.fit.fit_asr(dm1, 'p', iter=20000, burn=10000, thin=10)  # expect this to take around 1 hour

# <codecell>

# Make stage-1 prediction (to use as empirical prior in stage 2)
pred1 = dismod3.covariates.predict_for(dm1, dm1.parameters['p'], 'all', 'total', 'all',
                                      predict_region, predict_sex, predict_year, False, dm1.vars['p'], 0, 1)

# <codecell>

# re-load HCV model, filter to include only relevant data
dm = dismod3.data.load(model_path)
if predict_year == 2005:
    dm.keep(areas=[predict_region], sexes=['total', predict_sex], start_year=1997)
elif predict_year == 1990:
    dm.keep(areas=[predict_region], sexes=['total', predict_sex], end_year=1997)
else:
    raise Error, 'predict_year must equal 1990 or 2005'

# <codecell>

# set empirical priors based on results of stage-1 model
t = 'p'
for n, col in zip(dm1.vars[t]['beta'], dm1.vars[t]['X'].columns):
    stats = n.stats()
    dm.parameters[t]['fixed_effects'][col] = dict(dist='Constant', mu=stats['mean'], sigma=stats['standard deviation'])

for n, col in zip(dm1.vars[t]['alpha'], dm1.vars[t]['U'].columns):
    stats = n.stats()
    dm.parameters[t]['random_effects'][col] = dict(dist='Constant', mu=stats['mean'], sigma=stats['standard deviation'])

for n in dm1.vars[t]['sigma_alpha']:
    stats = n.stats()
    dm.parameters[t]['random_effects'][n.__name__] = dict(dist='TruncatedNormal', mu=stats['mean'], sigma=stats['standard deviation'], lower=.01, upper=.5)

# shift random effects to make REs for observed children of predict area have mean zero
re_mean = mean([dm.parameters[t]['random_effects'][area]['mu'] \
                   for area in dm.hierarchy.neighbors(predict_region) \
                   if area in dm.parameters[t]['random_effects']])
for area in dm.hierarchy.neighbors(predict_region):
    if area in dm.parameters[t]['random_effects']:
        dm.parameters[t]['random_effects'][area]['mu'] -= re_mean

# <codecell>

# calculate mean and standard deviation for age-pattern
emp_prior_mean = pred1.mean(0)

N,A = pred1.shape  # N samples, for A age groups
delta_trace = transpose([exp(dm1.vars['p']['eta'].trace()) for _ in range(A)])  # shape delta matrix to match prediction matrix
emp_prior_std = sqrt(pred1.var(0) + (pred1**2 / delta_trace).mean(0))

# <codecell>

# Fit stage-2 model
dm.vars += dismod3.ism.age_specific_rate(dm, 'p', predict_region, predict_sex, predict_year, mu_age_parent=emp_prior_mean, sigma_age_parent=emp_prior_std)
%time dismod3.fit.fit_asr(dm, 'p', iter=20000, burn=10000, thin=10)

# <codecell>

pred = dismod3.covariates.predict_for(dm, dm.parameters['p'],
                                      predict_region, predict_sex, predict_year,
                                      predict_region, predict_sex, predict_year, True, dm.vars['p'], 0, 1)

# <codecell>

dismod3.graphics.plot_data_bars(dm.get_data('p'))
import pymc as mc
plot(transpose([mc.rnormal(arange(101), .5**-2) for _ in range(1000)]), transpose(pred), 'k,', alpha=.1)
plot(pred.mean(0), color='w', linewidth=5)
plot(pred.mean(0), color='k', linewidth=3, label='Re-run mean')
legend()
axis(xmin=-5, xmax=105, ymin=-.01)

# <codecell>


