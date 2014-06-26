# Python implementation of CLEAR MOT multi target tracking evaluation metrics

- works for arbitrary dimensional data
- described in:
Keni, Bernardin, and Stiefelhagen Rainer. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing 2008 (2008).

## Requirements

    $ pip install numpy munkres  

or

    $ pip install -r requirements.txt
    
## Usage

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
    clear = clearmetrics.ClearMetrics(groundtruth, measurements, 1.5)
    clear.match_sequence()
    evaluation = [clear.get_mota(),
                  clear.get_motp(),
                  clear.get_fn_count(),
                  clear.get_fp_count(),
                  clear.get_mismatches_count(),
                  clear.get_object_count(),
                  clear.get_matches_count()]

Extended sample is in `example.py`.

