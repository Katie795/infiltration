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
        #elif my_data[i] >= 9000:
        #    my_data  = np.delete(my_data, i)
        #    x = np.delete(x, i)
        #    y = np.delete(y, i)
        #    pass
        else:
            i+=1


    d = {'x_coords': x, 'y_coords': y, 'height': my_data}

    df = pd.DataFrame(data=d)
    df = df[df['height'].notna()]
    print(df)
    df.to_csv(fileoutputname, header=True, index=False)  

def find_rain_garden(filename, fileoutputname):
    x, y = generate_coords(filename)

    my_data = np.genfromtxt(filename, delimiter=',')

    my_data = my_data.flatten()
    my_data_new = my_data.flatten()
    x = x.flatten()
    y = y.flatten()
    x_new = []
    y_new = []
    garden_height = []
    
    i=0
    x1, y1 = 14.6, 3.8
    x2, y2 = 47.2, 52.6
    x3, y3 = 52.2, 50.2
    x4, y4 = 20, 1.8

    while i < len(my_data):

        bounding_box = [y[i] - y1 >= (y4 - y1)/(x4 - x1) * (x[i] - x1),
                        y[i] - y1 <= (y2 - y1)/(x2 - x1) * (x[i] - x1),
                        y[i] - y3 <= (y2 - y3)/(x2 - x3) * (x[i] - x3),
                        y[i] - y4 >= (y4 - y3)/(x4 - x3) * (x[i] - x4)]

        if all(bounding_box) and my_data[i] <= -9000:
            x_new.append(x[i])
            y_new.append(y[i])
            my_data_new[i] = -2

        i += 1
    d = {'x_coords': x_new, 'y_coords': y_new, 'height': garden_height}

    df = pd.DataFrame(data=d)
    df = df[df['height'].notna()]
    print(df)
    df.to_csv(fileoutputname, header=True, index=False)


def add_rain_garden_to_domain(gardenfilename, roadfilename):

    df_garden = pd.read_csv(gardenfilename)
    df_road = pd.read_csv(roadfilename)

    full_df = pd.concat([df_garden, df_road], axis=0)
    full_df.to_csv("data/full_domain.csv", header=True, index=False)

def find_indices_rain_garden(filename, fileoutputname):

    my_data = np.genfromtxt(filename, delimiter=',')
    print(my_data)
    i=1
    while i < len(my_data):
        my_data[i, 0] = int(my_data[i, 0] / dx)
        my_data[i, 1] = int(my_data[i, 1] / dx)
        i+=1
        pass
    d = {'x_index': my_data[1:, 0], 'y_index': my_data[1:, 1], 'height': my_data[1:, 2]}
    df = pd.DataFrame(data=d)
    df["x_index"] = pd.to_numeric(df["x_index"], downcast="integer")
    df["y_index"] = pd.to_numeric(df["y_index"], downcast="integer")
    df.to_csv(fileoutputname, header=True, index=False)
    print(df)

find_indices_rain_garden(f"data/rain_garden.csv", "data/rain_garden_indices.csv")

find_rain_garden(f"results/.csv", "results/rain_garden.csv")

add_rain_garden_to_domain("data/rain_garden.csv", "results/road_10.csv")


#for i in range(1,11):
#    concatonate_csv_files(f"results/h_road_{100*i}.csv", f"results/road_{i}.csv")
    #concatonate_csv_files(f"results/h_garden_{100*i}.csv", f"results/garden_{i}.csv")

#for i in range(1,10):
#    concatonate_csv_files(f"results/h_{10*i}.csv", f"results/h_full_{10*i}.csv")