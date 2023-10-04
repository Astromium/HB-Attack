
import pickle

def parse_line(line):
    # function that takes a line as a string and returns the indices of features present in that line (constraint)
    chars = ['y_', '-', '>=', '0', '+']
    for char in chars:
        line = line.replace(char, '')
    line = line.strip()
    line = line.split(' ')
    return [int(a) for a in line if a != '']

filename = "./botnet_constraints.txt"

file = open(filename, 'r')
lines = file.readlines()

constraints = []

for line in lines:
    g = parse_line(line)
    print(f'g : {g}')
    constraints.append(g)

with open('./constraints_list_botnet.pkl', 'wb') as f:
    pickle.dump(constraints, f)
