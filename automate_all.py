import torch
import os
import random
import argparse
from generate import generate

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_generations', type=int, default=200, help="Number of times to generate text")
args = argparser.parse_args()

# get models 
def get_models():
    files = [f for f in os.listdir("./trained") if os.path.isfile(os.path.join("./trained", f))]
    models = []
    for file in files:
        model_path = os.path.join("./trained", file)
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.name = file
        model.eval()
        models.append(model)
    return models

# get input files 
def get_input():
    files = [f for f in os.listdir("./data") if os.path.isfile(os.path.join("./data", f))]
    input_files = []
    for file in files:
        file_path = os.path.join("./data", file)
        input_files.append(file_path)
    return input_files

# read input file
def read_input_file(input_file):
    with open(input_file, 'r') as f:
        text = f.read()
    return text.split()

# generate input sequence
def generate_prime_str(words, input_size):
    start_index = random.randint(0, len(words) - input_size)
    prime_str = ' '.join(words[start_index:start_index + input_size])
    return prime_str

# generate text from model using sequence of words
def generate_text(model, prime_str):
    return generate(model, prime_str)

# get cleaner file name
def get_file_name(model_name, input_file, size):
    model_name = model_name.split(".")[0]
    input_file = input_file.split("/")[2].split(".")[0]
    print(input_file)
    return "./results/" + model_name + "$" + input_file + "$" + str(size) + "e.txt"



def main():
    sizes = [1, 3, 5]
    models = get_models()
    input_files = get_input()

    for model in models:
        print(model.name)
        for input_file in input_files:
            words = read_input_file(input_file)
            for size in sizes:
                curr_output = get_file_name(model.name, input_file, size)

                with open(curr_output, 'w') as output_file:
                    for _ in range(args.num_generations):
                        prime_str = generate_prime_str(words, size)
                        generated_text = generate_text(model, prime_str)
                        output_file.write(f"{generated_text}\n\n")
                print(f"Generated text saved to {curr_output}")


if __name__ == "__main__":
    main()
