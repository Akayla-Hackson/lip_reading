def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        return "The file does not exist."


dataset_names = ['pretrain.txt', 'train.txt', 'val.txt', 'test.txt']
for file_name in dataset_names:
    line_count = count_lines_in_file(file_name)
    print(f"The file '{file_name}' contains {line_count} lines.")
