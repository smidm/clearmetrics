import clearmetrics

# 1d ground truth and measurements for 3 frames
groundtruth = {0: [2, 3, 6],
               1: [3, 2, 6],
               2: [4, 0, 6]
               }

measurements = {
    0: [1, 3, 8],
    1: [2, 3, None, 6],
    2: [0, 4, None, 6, 8]
}
print 'Groundtruth:'
print groundtruth
print '\nMeasurements:'
print measurements
print ''

clear = clearmetrics.ClearMetrics(groundtruth, measurements, 1.5)
clear.match_sequence()
for frame in groundtruth.keys():
    print 'Frame ' + str(frame) + ' matches:'
    for gt in clear.measurements_matches[frame]:
        if gt is not None and gt != -1:
            print 'gt: %d, m: %d, distance %f' % \
                (gt, clear.gt_matches[frame][gt], clear.gt_distances[frame][gt])
    print ''

evaluation = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]
print 'MOTA, MOTP, FN, FP, mismatches, objects, matches'
print evaluation              