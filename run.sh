# Step 1:
# Copy *.lab files from the original archive and round them
python gen_lab.py

# Step 2
# Perform segmentation.
python perf_segmentation.py

# Step 3:
# Make labels for training
# 1. time-lag model
# 2. duration model
# 3. acoustic model
python finalize_lab.py
