######## Counts the lines in a txt file ############
def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        return "The file does not exist."

# EX Usage: 
# dataset_names = ['pretrain.txt', 'train.txt', 'val.txt', 'test.txt']
# for file_name in dataset_names:
#     line_count = count_lines_in_file(file_name)
#     print(f"The file '{file_name}' contains {line_count} lines.")




######### Reads the ground truth txt file ############
def read_phrase_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    phrase = ""

    # Loop thru each line & check for the "Text:" label
    for line in lines:
        if "Text:" in line:
            # Split the line at "Text:" & strip any leading/trailing whitespace
            parts = line.split("Text:")
            if len(parts) > 1:
                phrase = parts[1].strip()  # Get the text after "Text:"
                break 

    return phrase
# EX Usage: 
# text_path = './LRS2/data_splits/pretrain/5535415699068794046/00020/00020.txt'
# text = read_phrase_from_file(text_path)
