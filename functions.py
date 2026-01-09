# Generate x and y coordinates for data visualisation
import numpy as np
import pandas as pd
dx = 0.2

def generate_coords(filename):

    my_data = np.genfromtxt(filename, delimiter=',')
    nx = len(my_data[0])
    ny = len(my_data)
    x_coords = np.zeros((ny, nx))
    y_coords = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny-1):
            x_coords[j][i] = dx * i
            y_coords[j][i] = dx * ny - dx * (j + 1)

    print(f"x_coords are {x_coords}")
    print(f"y_coords are {y_coords}")

    np.savetxt("results/x_coords.csv", x_coords, delimiter=",")
    np.savetxt("results/y_coords.csv", x_coords, delimiter=",")
    return x_coords, y_coords



def concatonate_csv_files(filename, fileoutputname):

    x, y = generate_coords(filename)

    my_data = np.genfromtxt(filename, delimiter=',')

    my_data = my_data.flatten()
    x = x.flatten()
    y = y.flatten()
    
    i = 0
    while i < len(my_data):
        if i > len(my_data):
            print("we have reached the end of the array")
            break
        elif my_data[i] <= -9000:
            my_data  = np.delete(my_data, i)
            x = np.delete(x, i)
            y = np.delete(y, i)
            pass
        elif my_data[i] >= 9000:
            my_data  = np.delete(my_data, i)
            x = np.delete(x, i)
            y = np.delete(y, i)
            pass
        else:
            i+=1


    d = {'x_coords': x, 'y_coords': y, 'height': my_data}

    df = pd.DataFrame(data=d)
    df = df[df['height'].notna()]
    print(df)
    df.to_csv(fileoutputname, header=True, index=False)  


for i in range(1,11):
    concatonate_csv_files(f"results/h_road_{100*i}.csv", f"results/road_{i}.csv")
    concatonate_csv_files(f"results/h_garden_{100*i}.csv", f"results/garden_{i}.csv")

#for i in range(1,10):
#    concatonate_csv_files(f"results/h_{10*i}.csv", f"results/h_full_{10*i}.csv")