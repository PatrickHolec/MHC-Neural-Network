
import numpy as np
import matplotlib.pyplot as plt

def RepresentsInt(s):
    try: 
        int(s)
        if int(s) > 0: 
            return True
        else: 
            return False
    except ValueError:
        return False


files = ['round0.txt','round1.txt','round2.txt','round3.txt','round4.txt','round5.txt']

results = []
percentiles = []

for fname in files:
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    results.append([int(x) for x in content[1:] if RepresentsInt(x)])
    percentiles.append([float(i)/len(results[-1]) for i in xrange(1,len(results[-1])+1)])

# order shit
results = [sorted(r, key=int, reverse=True) for r in results] 
    
fig = plt.figure()
ax = plt.gca()
    
for x,y,i in zip(percentiles,results,xrange(len(results))):
    plt.plot(x, y, label = 'Round {}'.format(i))

ax.set_yscale('log')

plt.xlabel('Percentile (clonal)')
plt.ylabel('Read count')
plt.legend()

print 'here'
plt.show()    