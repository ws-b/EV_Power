import numpy as np
path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/60097.txt'

with open(path, 'r') as file:
    data = file.read().splitlines()

# remove first item from first line
data[0] = ','.join(data[0].split(',')[1:])

output = []
for line in data:
    items = line.split(',')
    while len(items) >= 3:
        output.append(items[:3])
        items = items[3:]
    if len(items) > 0:
        output.append(items + ['']*(3-len(items)))

# create a NumPy array from the output list
arr = np.array(output)

# write the array to a CSV file
np.savetxt('/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/60097.csv', arr, delimiter=',', fmt='%s')
