# Common settings

from os.path import join, expanduser

# Output directory
# All the igenerated labels, intermediate files, and segmented wav files
# will be saved in the following directory
# Note that manually corrected files are managed by the following repository:
# https://github.com/r9y9/kiritan_singing_extra
out_dir = "nit_song070_extra"

# PLEASE CHANGE THE PATH BASED ON YOUR ENVIRONMENT
nit_song070_root = join(expanduser("~"), "HTS-demo_NIT-SONG070-F001")

# Song segmentation by silence durations.
# TODO: would be better to split songs by phrasal information in the musical scores

# Split song by silences (in sec)
segmentation_threshold = 0.4

# Min duration for a segment
# note: there could be some execptions (e.g., the last segment of a song)
segment_min_duration = 5.0

# Force split segments if long silence is found regardless of min_duration
force_split_threshold = 5.0


# Offset correction
# If True, offset is computed in an entire song
# otherwise offset is computed for each segment
global_offset_correction = False


file_indexes_to_be_processed = [3,4,7,10,12,14,15,16,19,20,21,22,23,25,28,29,30,37,39,40,41,45,48,50,51,54,55,59,60,63,70]
