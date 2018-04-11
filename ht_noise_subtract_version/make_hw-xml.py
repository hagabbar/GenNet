from minke import mdctools, distribution, sources

mdcset = mdctools.MDCSet(['H1'])

times = distribution.uniform_time(1126620016, 1136995216, number = 1000)

hrss_values = distribution.log_uniform(1e-20, 1e-20, len(times))

wnb = sources.WhiteNoiseBurst(duration=0.1, bandwidth=10, frequency=1000,
                              hrss=1e-23, time=1126630000, seed=3)

mdcset + wnb

for hrss, time in zip(hrss_values, times):
    wnb = sources.WhiteNoiseBurst(duration=0.1, bandwidth=10, frequency=1000, 
                                  hrss=hrss, time=time, seed=3)
    mdcset + wnb
mdcset.save_xml('wnb1000b10tau0d1.xml.gz')

