import random as rand
import csv

#6 numbers are randomly generated. Each number is randomly generated between 1 and 10.
#The average of the numbers is calculated. If it is above 5, result is set to 1. Otherwise,
#result is set to 0.
#the output is a list with the 6 numbers and result in that order.
def generate_line():
    total = 0
    output = []
    
    for i in range(6):
        num = rand.randint(1, 10)
        total += num
        output.append(num)

    #calculate average and determine result
    average = total /6
    if (average < 5):
        output.append(0)
    else:
        output.append(1)

    return output



def main():
    size = int(input("enter the number of data points to create: "))
    with open("test_data.csv", "w", newline="") as outfile:
        a = csv.writer(outfile)
        a.writerow([size, 6])
        for i in range(size):
            line = generate_line()
            a.writerow(line)

main()