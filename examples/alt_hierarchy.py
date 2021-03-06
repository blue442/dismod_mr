# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Setting the hierarchy in DisMod-MR
#
# The goal of this document is to demonstrate how to set the spatial hierarchy for the random effects in DisMod-MR.
#
# The examples are based on a spatial hierarchy of Japan, provided by Ver Bilano, and included in the examples directory.

# <codecell>

import random
import dismod_mr
import pandas as pd
import numpy as np

# <codecell>

df = pd.read_csv('./examples/hierarchy.csv')

# <codecell>

df.head()

# <markdowncell>

# First, we will use simulation to generate $n$ rows of input data.

# <codecell>


# <codecell>

n = 100

dm = dismod_mr.data.ModelData()
inp = pd.DataFrame(columns=dm.input_data.columns, index=range(n))

# data type, value, and uncertainty
inp.data_type = 'p'
inp.value = .5 + .1*np.random.randn(n)
inp.effective_sample_size = 1000.

# geographic information (to be used for random effects)
inp.area = [random.choice(df.Prefecture) for i in range(n)]

inp.sex = 'total'
inp.age_start = 50
inp.age_end = 50

inp.standard_error = np.nan
inp.upper_ci = np.nan
inp.lower_ci = np.nan

# put data in model
dm.input_data = inp

# <codecell>

# set model parameters for simple fit
dm.parameters['p'] = {'level_value': {'age_after': 100, 'age_before': 1, 'value': 0.},
                      'parameter_age_mesh': [0, 100]}

# <markdowncell>

# The following code generates a single level hierarchy, with all prefectures below the national level:

# <codecell>

for p in df.Prefecture:
    dm.hierarchy.add_edge('all', p)

# <markdowncell>

# That is all there is to it!

# <codecell>

dm.vars = dismod_mr.model.asr(model=dm, data_type='p', rate_type='neg_binom')
%time dismod_mr.fit.asr(model=dm, data_type='p', iter=10000, burn=5000, thin=5)

# <codecell>

dismod_mr.plot.effects(model=dm, data_type='p', figsize=(18, 10))

# <markdowncell>

# To use a two-level hierarchy instead, simply build the regions into the hierarchy graph:

# <codecell>

dm = dismod_mr.data.ModelData()
dm.input_data = inp
dm.parameters['p'] = {'level_value': {'age_after': 100, 'age_before': 1, 'value': 0.},
                      'parameter_age_mesh': [0, 100]}

# <codecell>

for i, row in df.iterrows():
    dm.hierarchy.add_edge('all', row['Region'])
    dm.hierarchy.add_edge(row['Region'], row['Prefecture'])

# <codecell>

dm.vars = dismod_mr.model.asr(model=dm, data_type='p', rate_type='neg_binom')
%time dismod_mr.fit.asr(model=dm, data_type='p', iter=10000, burn=5000, thin=5)

# <codecell>

dismod_mr.plot.effects(model=dm, data_type='p', figsize=(18, 14))

# <markdowncell>

# It would be great to extend this example so that the results differed in a meaningful way when using one- and two-level hierarchical models.  This is left as an exercise to the reader.

# <codecell>

# !date
