from minke import mdctools, distribution, sources
from random import randint

mdcset = mdctools.MDCSet(['H1'])

times = distribution.uniform_time(1126620016, 1136995216, number = 1000)

hrss_values = distribution.log_uniform(1e-23, 1e-23, len(times))

sineGauss = sources.SineGaussian(q=15, frequency=100, polarisation='linear',
                                 hrss=1e-23, time=1126630000, seed=3)

mdcset + sineGauss

for hrss, time in zip(hrss_values, times):
    sineGauss = sources.SineGaussian(q=15, frequency=randint(100,200), polarisation='linear',
                                     hrss=hrss, time=time, seed=3)
    mdcset + sineGauss
mdcset.save_xml('sineGauss100b10tau0d1.xml.gz')

