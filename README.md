DATA SETUP STEPS:
1) To download dataset go to: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html
2) Enter username & password provided (must submit agreement first to then receive login info)
3) Download all files (including the .txt files)
4) Run the extract_all_data.py
      -- This will:
           1) Concatinate the downloaded files into a .tar file
           2) Extract all the data from the .tar file (this creates 'main' and 'pretrain' folders
           3) Separates the data into the designated pretrain, train, val, test splits (based off of the given .txt files)
           4) Loops thru the data splits and produces the frames for each mp4 file in the dataset (takes a longg time)

