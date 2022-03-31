import numpy as np

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    
if __name__ == '__main__':
    intervals = list(np.linspace(-0.5, 0.5, 11))
    reachable_interval = [-0.3, 0.3]
    bin_intervals = [[intervals[i], intervals[i+1]] for i in range(len(intervals)-1)]
    pts_to_test = []
    print(bin_intervals)
    
    for i in range(len(bin_intervals) - 1):

        for j in range(len(bin_intervals) - 1):

            for k in range(len(bin_intervals) -1):

                pts_to_test.append([min(abs(bin_intervals[i][0]), abs(bin_intervals[i][1])), min(abs(bin_intervals[j][0]), abs(bin_intervals[j][1])), min(abs(bin_intervals[k][0]), abs(bin_intervals[k][1]))])

    count_cpt = 0
    for pt in pts_to_test:
        if pt[0]**2 + pt[1]**2 + pt[2]**2 < 0.3**2:
            count_cpt += 1

    print("Number of bins reachable: ", count_cpt)
