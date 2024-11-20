# What Was That?: Lip-Reading Silent Videos

## Authors: Akayla Hackson, Miguel Gerena, Linyin Lyu
**Stanford University**

**Contact**: [akayla@stanford.edu](mailto:akayla@stanford.edu), [miguelg2@stanford.edu](mailto:miguelg2@stanford.edu), [llyu@stanford.edu](mailto:llyu@stanford.edu)

---

### Abstract
This study develops three models to advance lip-reading technology using silent video analysis, achieving different levels of success in recognizing spoken words and sentences purely from visual information.

---

### Introduction
Lip-reading, or visual speech recognition, involves transcribing text from silent videos. It has applications in diverse fields such as surveillance, silent video conferencing, and archival film transcription. The task faces challenges due to visual ambiguities among similar phonetic sounds and diverse mouth movements across speakers.

---

### Model Development
- **Speaking Detection Model**: Uses a 3D-CNN to detect speaking activity with 77.63% accuracy.
- **Word Prediction Model (LRW)**: Extends the 3D-CNN with LSTM to handle temporal dependencies, focusing on single-word prediction with 52.19% accuracy.
- **Sentence Prediction Model**: Combines the previous architectures with an Encoder-Decoder Transformer layer aimed at constructing full sentences, which needs further improvement due to issues with local minima.

---

### Methodology
Our approach combines spatiotemporal CNNs for feature extraction from video frames and Bi-LSTM for modeling temporal sequence dynamics. The models are further enhanced by transformer layers for handling longer-range dependencies and improving sentence structure prediction.

---

### Datasets and Preprocessing
- **LRS2 Dataset**: Used for predicting both speaking status and full sentences.
- **LRW Dataset**: Employed for single-word predictions.
- Videos were processed to maintain uniform frame rates and normalized for consistent input to the models.

---

### Experiments and Results
Initial experiments on speaking detection provided a foundation with reasonable accuracy. The single-word prediction model showed promise but highlighted the complexity of extending to full sentences. The most advanced model, intended for full sentences, demonstrated the need for significant refinement.

---

### Conclusions and Future Work
While the models for detecting speech and predicting words performed commendably, the sentence prediction model requires further development. Future work will focus on refining the models using additional data, extended training, and potentially incorporating more advanced techniques like self-supervised learning to handle the complexities of lip reading.

---

### Acknowledgements
We thank our colleagues and department for their support and the insightful discussions that helped shape this research.

---




Lip reading, also known as visual speech recognition,
involves transcribing text from videos. This is an ability to
recognize what is being said from visual information alone.
Lip reading is inherently ambiguous as different characters
could produce exactly the same lip sequence (e.g. ‘p’ and
‘b’) and different people could say the same word but have
different lip movements.
In this study, we developed three models aimed at advancing lip reading technology. The first model focused
on being able to detect if the subject was speaking using a
3d-CNN without any addition for handling long-range temporal dependencies. This first model successfully predicts
whether a person is speaking with an accuracy of 77.63%,
serving as a strong foundation for our project. The second
model used this CNN network to produce characters and
forming the words being spoken. It handled short and long
term dependencies utilizing an LSTM. It is focused on the
LRW dataset and achieves a promising accuracy of 52.19%,
indicating the potential to accurately predict spoken words
without audio input. Our third model used the previous architectures with an Encoder-Decoder Transformer layer on
top in order to handle the word dependencies and able to establish word and sentence structure. This third model, designed to predict entire spoken sentences, requires further
improvement as it currently struggles with local minima issues. Although the first two models showed commendable
performance, the complexities of sentence prediction necessitate continued experimentation. Due to time constraints,
we were unable to fully address these challenges within this
project, but we are committed to refining our approach and
enhancing the model’s performance in future work.
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

