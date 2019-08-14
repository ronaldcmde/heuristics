from os import listdir
import csv
from os.path import isfile, join

mypath = "resources/"

def main():
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles: 
        with open(mypath + file) as file:
            line = file.readline().split('\t')
            n = int(line[0])
            m = int(line[1])
            Q = v = []
            for i in range(0, m):
                type = file.readline().split('\t')
                index = int(type[0])
                quantity = int(type[1])
                Qk = int(type[2])
                Vk = float(type[3].replace(',', '.'))

                for k in range(0, quantity):
                    Q.append(Qk)
                    v.append(Vk)
            
            (indexes, x, y, q) = zip(*[line for line in csv.reader(file, delimiter='\t')])

                
            # for line in csv.reader(file, delimiter="\t"):


        
        


if __name__ == "__main__": 
    main()