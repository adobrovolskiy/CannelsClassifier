import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,20))


ax = fig.add_subplot(111)

## the data
N = 5
menMeans = [0.18, 0.35, 0.30, 0.35, 0.27]


## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.3                    # the width of the bars

## the bars
rects1 = ax.bar(ind, menMeans, width,
                color='black',
                error_kw=dict(elinewidth=2,ecolor='red'))


# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0, 1)
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
xTickMarks = ['Group'+str(i) for i in range(1,6)]
ax.set_xticks(ind)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)



# # add a legend
# ax.legend( (rects1[0], 'Men'))

plt.show()