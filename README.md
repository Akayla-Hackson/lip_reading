# DATA SETUP STEPS:
1. To download the dataset, visit: [LRS2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
2. Enter the username & password provided (you must submit an agreement first to then receive login info).
3. Download all files (including the .txt files) and store them in the LRS2 folder.
4. Run the `extract_all_data.py` script:
   - This script will:
     - Concatenate the downloaded files into a `.tar` file.
     - Extract all the data from the `.tar` file (this creates 'main' and 'pretrain' folders).
     - Separate the data into the designated pretrain, train, val, test splits (based on the given .txt files).
     - Loop through the data splits and produce the frames for each mp4 file in the dataset (this takes a long time).

# FAMILIARIZE YOURSELF WITH THE MODEL
1. Info:
   - `classes/dataset.py` holds the dataloader.
   - `classes/lip_reading.py` holds the main structure of the model (there are comments indicating the expected shapes of each layer's output, as well as print statements to help):
     - `classes/cnn.py` is the first layer.
     - `classes/lstm.py` is the second layer.
     - `classes/transformer.py` is the final layer (currently, it is only used as a decoder).
2. Running tests:
   - `classes/test_classes.py` contains scripts to test each of the files above:
     - The layer test cases use dummy data.
     - The complete model and the dataloader cases use the real LRS2 data.
     - Run using the command: `python -m classes.test_classes`.

# TRAINING
- `ak_main.py` contains the training loop:
  - Currently, only training is set up (validation is still needed).
  - See TensorBoard for tracking training progress.

