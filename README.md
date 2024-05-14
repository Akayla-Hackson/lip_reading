DATA SETUP STEPS:
1) To download dataset go to: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html
2) Enter username & password provided (must submit agreement first to then receive login info)
3) Download all files (including the .txt files) and store in LRS2 folder
4) Run the extract_all_data.py
      -- This will:
           1) Concatinate the downloaded files into a .tar file
           2) Extract all the data from the .tar file (this creates 'main' and 'pretrain' folders
           3) Separates the data into the designated pretrain, train, val, test splits (based off of the given .txt files)
           4) Loops thru the data splits and produces the frames for each mp4 file in the dataset (takes a longg time)

FAMILARIZE YOURSELF WITH THE MODEL
Info:
      - classes/dataset.py holds the dataloader
      - classes/lip_readying.py holds the main structure of the model (there are comments saying what the expected shapes of each layer output are, as well as print statements to help)
            -- classes/cnn.py is the first layer
            -- classes/lstm.py is the 2nd layer
            -- classes/transformer.py is the final layer (right not it is only used as a decoder)
Running tests: 
      - classes/test_classes.py holds script to test each of the files above.
            -- The layer test cases use dummy data
            -- The complete model and the dataloader cases use the real LRS2 data
            -- Run using command: python -m classes.test_classes


TRAINING
      - ak_main.py holds the training loop
            -- Only training is set up (validation is still needed)
            -- See Tensorboard for tracking training progress