import numpy as np
from hungarian import Hungarian
import sys


class ClearMetrics(object):
    """
    CLEAR multi target tracking metric evaluation.

    For arbitrary dimensional data.

    described in:
    Keni, Bernardin, and Stiefelhagen Rainer. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
    EURASIP Journal on Image and Video Processing 2008 (2008).

    Usage:

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
    clear = ClearMetrics(groundtruth, measurements, 1.5)
    clear.match_sequence()
    evaluation = [clear.get_mota(),
                  clear.get_motp(),
                  clear.get_fn_count(),
                  clear.get_fp_count(),
                  clear.get_mismatches_count(),
                  clear.get_object_count(),
                  clear.get_matches_count()]
    """

    def __init__(self, groundtruth, measurements, thresh):
        """
        Initialize ClearMetrics with input data.

        @param groundtruth:     [frame nr]    [target nr]
                                dict/list     list        ndarray, shape=(n,) or number or None
                                                          - ndarray for input data with dimensionality n
                                                          - number for 1D input data
                                                          - None means target is not present

        @param measurements:    [frame nr]    [target nr]
                                dict/list     list        ndarray, shape=(n,) or number or None

        @param thresh: float, maximum distance of a measurement from ground truth to be considered as true positive
        """
        self.groundtruth = groundtruth
        self.measurements = measurements
        self.thresh = thresh

        # following members hold evaluation results:

        # [frame nr]    [target nr]
        # dict/list     list        int (groundtruth or measurement index, -1 if no match)
        self.measurements_matches = None
        self.gt_matches = None
        self.gt_distances = None

        # {mismatches: {frame: idx}, fp: {frame: idx}, fn: {frame: idx}}
        self._problematic_frames = {'mismatches': {}, 'fp': {}, 'fn': {}}
        self._methods_processed = {
            'get_fp_count': False,
            'get_fn_count': False,
            'get_mismatches_count': False
        }

    def match_sequence(self):
        """
        Evaluate the sequence.

        Writes results to
            self.measurements_matches
            self.gt_matches
            self.gt_distances
        """
        prev_gt_matches = [-1] * len(self.groundtruth.values()[0])
        self.gt_matches = {}
        self.gt_distances = {}
        self.measurements_matches = {}
        for frame in sorted(self.groundtruth.keys()):
            if frame >= len(self.measurements):
                break
            self.gt_matches[frame], self.gt_distances[frame], self.measurements_matches[frame] = \
                self._match_frame(frame, prev_gt_matches)
            prev_gt_matches = self.gt_matches[frame]

    def get_problematic_frames(self):
        if not self._methods_processed['get_fp_count']:
            self.get_fp_count()
        if not self._methods_processed['get_fn_count']:
            self.get_fn_count()
        if not self._methods_processed['get_mismatches_count']:
            self.get_mismatches_count()

        return self._problematic_frames

    def get_fp_count(self):
        """
        Return number of false positives in the sequence.

        @return: FP count
        @rtype: int
        """
        # prevents duplicate entries when user for some reason
        # calls get_fp_count multiple times.
        called_first_time = self._methods_processed['get_fp_count']

        count = 0
        for frame in self.measurements_matches:
            matches = self.measurements_matches[frame]
            indexes = _indexes(matches, -1)
            count += len(indexes)
            if len(indexes) > 0 and called_first_time:
                self._problematic_frames['fp'][frame] = indexes

        self._methods_processed['get_fp_count'] = True
        return count

    def get_fn_count(self):
        """
        Return number of false negatives in the sequence.

        @return: FN count
        @rtype: int
        """
        # prevents duplicate entries when user for some reason
        # calls get_fn_count multiple times.
        called_first_time = self._methods_processed['get_fn_count']

        count = 0
        for frame in self.gt_matches:
            matches = self.gt_matches[frame]
            indexes = _indexes(matches, - 1)
            count += len(indexes)
            if len(indexes) > 0 and called_first_time:
                self._problematic_frames['fn'][frame] = indexes

        self._methods_processed['get_fn_count'] = True
        return count

    def get_mismatches_count(self):
        """
        Return number of identity mismatches.

        One mismatch occurs when measurement id assigned to a gt id changes.
        E.g. identity swap in one frame equals 2 identity mismatches.

        @return: number of mismatches in the sequence
        @rtype: int
        """
        # prevents duplicate entries when user for some reason
        # calls get_mismatches_count multiple times.
        called_first_time = self._methods_processed['get_mismatches_count']

        frames = sorted(self.groundtruth.keys())
        last_matches = np.array(self.gt_matches[frames[0]])
        mismatches = 0
        for frame in frames[1:]:
            if frame >= len(self.measurements):
                break
            matches = np.array(self.gt_matches[frame])
            mask_match_in_both_frames = (matches != -1) & (last_matches != -1)
            _test = matches[mask_match_in_both_frames] != last_matches[mask_match_in_both_frames]
            num = np.count_nonzero(_test)
            mismatches += num
            if num > 0 and called_first_time:
                indexes = _indexes((_test != 0).tolist(), True)
                self._problematic_frames['mismatches'][frame] = indexes

            last_matches = matches

        self._methods_processed['get_mismatches_count'] = True

        return mismatches

    def get_object_count(self):
        """
        Return number of ground truth objects in all frames in the sequence.

        @return: number of gt objects
        @rtype: int
        """
        object_count = 0
        for frame in sorted(self.groundtruth.keys()):
            if frame >= len(self.measurements):
                break
            targets = self.groundtruth[frame]
            object_count += len(targets) - targets.count(None)  # TODO np.array([]) empty arrays?
        return object_count

    def get_matches_count(self):
        """
        Return number of matches between ground truth and measurements in all frames in the sequence.

        @return: number of matches
        @rtype: int
        """
        distances = np.array([dists for dists in self.gt_distances.values()])
        matches_mask = distances != -1
        return distances[matches_mask].size

    def get_motp(self):
        """
        Return CLEAR MOTP score.

        MOTP is mean distance to ground truth / mean error. Lower is better.

        @return: MOTP score
        @rtype: float
        """
        distances = np.array([dists for dists in self.gt_distances.values()])
        matches_mask = distances != -1
        return distances[matches_mask].mean()

    def get_mota(self):
        """
        Return CLEAR MOTA score.

        Can be roughly understood as a ratio of correctly tracked objects. Bigger / closer to 1 is better.

        @return: MOTA score, <= 1
        @rtype: float
        """
        return 1 - (self.get_fp_count() + self.get_fn_count() + self.get_mismatches_count()) / \
               float(self.get_object_count())

    def _get_sq_distance_matrix(self, frame):
        """
        Compute squared distances between ground truth and measurements objects.

        Distance is sys.maxint when gt or measurement is not defined (None).

        @param frame: frame number
        @type frame: int
        @return: distance matrix (not symmetric!)
        @rtype: np.ndarray, shape=num ground truth, num measurements
        """
        n_gt = len(self.groundtruth[frame])
        n_meas = len(self.measurements[frame])
        distance_mat = np.zeros((n_gt, n_meas))
        for i in xrange(n_gt):
            gt_pos = self.groundtruth[frame][i]
            for j in xrange(n_meas):
                measured_pos = self.measurements[frame][j]
                if gt_pos is None or measured_pos is None:
                    distance_mat[i, j] = sys.maxint
                else:
                    distance_mat[i, j] = np.sum((measured_pos - gt_pos) ** 2)
        return distance_mat

    def _match_frame(self, frame, prev_gt_matches):
        """

        @type frame: int
        @type prev_gt_matches: list
        @return: gt_matches - list of measurement ids to that the ground truth objects match
                              None for gt objects not present in the frame
                 gt_distances - distances from ground truth objects to matched measured objects
                                None for objects not found in the frame
                 measurements_matches - list of ground truth ids to that the measured objects match
        @rtype: list, list, list
        """
        sq_distance = self._get_sq_distance_matrix(frame)
        sq_distance[sq_distance > (self.thresh ** 2)] = sys.maxint

        # set all ground truth matches to FN or not defined
        gt_matches = []
        for i in xrange(len(self.groundtruth[frame])):
            if self.groundtruth[frame][i] is None:
                gt_matches.append(None)
            else:
                gt_matches.append(-1)
        # set all measurements matches to FP or not defined
        gt_distances = [-1] * len(self.groundtruth[frame])
        measurements_matches = []
        for i in xrange(len(self.measurements[frame])):
            if self.measurements[frame][i] is None:
                measurements_matches.append(None)
            else:
                measurements_matches.append(-1)

        # verify TP from previous frame
        for prev_gt, prev_measurement in enumerate(prev_gt_matches):
            if prev_measurement != -1 and sq_distance[prev_gt, prev_measurement] <= (self.thresh ** 2):
                gt_matches[prev_gt] = prev_measurement
                measurements_matches[prev_measurement] = prev_gt
                gt_distances[prev_gt] = np.sqrt(sq_distance[prev_gt, prev_measurement])
                # prev_gt and prev_measurement are excluded from further matching
                sq_distance[prev_gt, :] = sys.maxint
                sq_distance[:, prev_measurement] = sys.maxint

        hungarian = Hungarian(sq_distance)
        hungarian.calculate()
        matches = hungarian.get_results()

        # fill in new TP
        for m in matches:
            if sq_distance[m[0], m[1]] == sys.maxint:
                continue
            gt_matches[m[0]] = m[1]
            measurements_matches[m[1]] = m[0]
            gt_distances[m[0]] = np.sqrt(sq_distance[m[0], m[1]])

        return gt_matches, gt_distances, measurements_matches


def _indexes(l, el):
    """returns all indexes where elements of l equals el """
    return [i for i, x in enumerate(l) if x == el]
