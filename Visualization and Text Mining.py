"""
Apply regular expression to clean data

Apply ML/DL models for sentiment analysis

"""

import pandas as pd
import numpy as np
import re
import os
from matplotlib import pyplot as plt
import seaborn as sns


#set the format for console display
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_colwidth", 20)



#data preprocessing
missing_value = ['N.A.',"--", "na","n/a"]
path = "/Users/guanyanghu/SMU/PythonForDataScience/exp_tweet_week9.csv"
test = pd.read_csv(path, encoding='ISO-8859-1', skiprows=1, na_values=missing_value)

"""
Data Sample
        user_id  # of followers                                               text
272    39612780            1126  News from Wired: New 'Super Smash Bros. Ultima...
996     1981691           17912  #indian #youtube #video marketing should be sh...
995   229726308            3826  RT @daisyyfields: wow the contestant going aga...
943   613454359            2437               I can save up to RM100 every month! 
450    98570596             464  RT @nayoxx: Listen.;1. The amount of active Ar...
394  1289009508             476  RT @aliaassabrina: Abang, adik rindu9þ69þ69...
600  1729547066            1059  RT @ddu_boo: 181102 595Ý5 234Ý53321;...
618  2208541680            2606  RT @KleinmanEnergy: What we're reading: Carlyl...
471  2222961974             466  RT @rafiqrohizad: Aku harap aku sempat untuk a...
675    14073875            7107  @jemimaskelley @sarhlou Couldn¡¯t notice you h...

"""

test['user_id'] = test['user_id'].apply(lambda x: int(x))
test['# of followers'] = test['# of followers'] .apply(lambda x: int(x) if str(x).isnumeric() else int(0))
median = test['# of followers'].median()
test['# of followers'].replace(0, median, inplace=True)

p = re.compile(r'RT\s*@(?P<target>[^:]*):')
test['retweet_target'] = test['text'].apply(lambda x: p.match(str(x)).group('target') if p.match(str(x)) else " ")


mmin = min(test["# of followers"])
mmax = max(test["# of followers"])
mmedian = test["# of followers"].median()
print(mmin)
print(mmax)
print(mmedian)

# values = pd.Series([np.percentile(test['# of followers'], i) for i in range(5,100,5)])
# values.plot.bar()
# plt.show()


bins = [mmin, 2085, 95714,  mmax]
labels =['low','median','high']
test['popularity_bin'] = pd.cut(test['# of followers'], bins,labels=labels)


test['text']=test['text'].apply(str)
test['char_count'] = test['text'].apply(len)



test['avg_word'] = test['text'].apply(lambda x: len(str(x))/len(str(x).split(" ")))



print(test.head(10))

