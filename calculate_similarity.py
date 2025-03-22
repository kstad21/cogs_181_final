import sys
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

def append_to_csv(csv_filename, trained_original, input_original, input_size, generation_num, similarity_1, similarity_2):
    file_exists = False
    try:
        with open(csv_filename, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['trained_original', 'input_original', 'input_size', 'generation_num', 'similarity_to_trained', 'similarity_to_input']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'trained_original': trained_original,
            'input_original': input_original,
            'input_size': input_size,
            'generation_num': generation_num,
            'similarity_to_trained': similarity_1,
            'similarity_to_input': similarity_2
        })

def get_model_input_size(file_path):
    to_return = []
    splits = file_path.split("$")
    model = splits[0]
    input = splits[1]
    size = splits[2].split(".")[0]

    to_return.append(model)
    to_return.append(input)
    to_return.append(size)
    return to_return

def get_cross_files():
    files = [f for f in os.listdir("./results") if os.path.isfile(os.path.join("./results", f))]
    files = [f for f in files if "e.txt" in f]
    return files

def get_original_text(name):
    files = [f for f in os.listdir("./data") if os.path.isfile(os.path.join("./data", f))]
    for file in files:
        if name in file:
            return file
    print("UH OH")
    return ""
    

def main():
    files = get_cross_files()
    for file in files:
        info = get_model_input_size(file)
        print(info)
        
        cross_gen = read_file("./results/" + file)
        original_model = get_original_text(info[0])
        original_input = get_original_text(info[1])
        size = info[2]
        og_model_text = read_file("./data/" + original_model)
        og_input_text = read_file("./data/" + original_input)

        cross_model = calculate_cosine_similarity(cross_gen, og_model_text)
        cross_input = calculate_cosine_similarity(cross_gen, og_input_text)

        append_to_csv("similarity_results.csv", original_model, original_input, size, 200, cross_model, cross_input)


if __name__ == "__main__":
    main()