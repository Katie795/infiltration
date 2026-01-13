# Functions for pre- and post-processing data
import numpy as np
import pandas as pd
dx = 0.2

def generate_coords(filename):
    """Generate csv files containing the x and y coordinate data for a uniform grid
    
    :args filename: csv file to generate coordintates from"""

    my_data = np.genfromtxt(filename, delimiter=',')
    nx = len(my_data[0])
    ny = len(my_data)
    x_coords = np.zeros((ny, nx))
    y_coords = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny-1):
            x_coords[j][i] = dx * i
            y_coords[j][i] = dx * ny - dx * (j + 1)

    np.savetxt("data/x_coords.csv", x_coords, delimiter=",")
    np.savetxt("data/y_coords.csv", y_coords, delimiter=",")
    return x_coords, y_coords

def create_paraview_table(x_file, y_file, z_file, output_file):
    """Write a paraview-readable csv file with coordinate data and height data, removing data not in
    domain of interest

    :args x_file: file with the x coordinate data
    :args y_file: file with the y coordinate data
    :args z_file: file with the z coordinate data
    :args output_file: name of the file to write output data to"""
    
    x,y = generate_coords(z_file)
    #x = np.genfromtxt(x_file, delimiter=',')
    #y = np.genfromtxt(y_file, delimiter=',')
    my_data = np.genfromtxt(z_file, delimiter=',')

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
        else:
            i+=1


    d = {'x_coords': x, 'y_coords': y, 'height': my_data}

    df = pd.DataFrame(data=d)
    df = df[df['height'].notna()]
    df.to_csv(output_file, header=True, index=False)  

def find_rain_garden(x_file, y_file, z_file, output_file):
    """Find the coordinates of the points within the rain garden and save as csv file
    
    :args x_file: file with the x coordinate data
    :args y_file: file with the y coordinate data
    :args z_file: file with the z coordinate data
    :args output_file: name of the file to write output data to"""
    x = np.genfromtxt(x_file, delimiter=',')
    y = np.genfromtxt(y_file, delimiter=',')

    my_data = np.genfromtxt(z_file, delimiter=',')

    my_data = my_data.flatten()
    my_data_new = my_data.flatten()
    x = x.flatten()
    y = y.flatten()
    x_new = []
    y_new = []
    garden_height = []
    
    i=0
    x1, y1 = 14.6, 3.8 # coordinates of a bounding box around the rain garden.
    x2, y2 = 47.2, 52.6 # the coordinates not on the road have been given a 
    x3, y3 = 52.2, 50.2 # value of -9999, so we search for these values
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
    df.to_csv(output_file, header=True, index=False)


def add_rain_garden_to_domain(garden_file, road_file):
    """Concatonate the coordinates of the rain garden with the coordinates of the road
    
    :args garden_file: the file with the rain garden data
    :args road_file: the file with the road data"""

    df_garden = pd.read_csv(garden_file)
    df_road = pd.read_csv(road_file)

    full_df = pd.concat([df_garden, df_road], axis=0)
    full_df.to_csv("data/full_domain.csv", header=True, index=False)

def find_indices_rain_garden(input_file, output_file):
    """Convert coordinates to indices for the rain garden
    
    :args input_file: name of the input file
    :args output file: name of the output file"""

    my_data = np.genfromtxt(input_file, delimiter=',')
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
    df.to_csv(output_file, header=True, index=False)

#find_indices_rain_garden(f"data/rain_garden.csv", "data/rain_garden_indices.csv")
#find_rain_garden(f"results/.csv", "results/rain_garden.csv")
#add_rain_garden_to_domain("data/rain_garden.csv", "results/road_10.csv")

#create_paraview_table("data/x_coords.csv", "data/y_coords.csv", "results/h_road_10.csv", "results/road.csv")


def post_processing(file, name, i):
    """Function to post-process the csv files into a paraview readable format"""
    create_paraview_table("data/x_coords.csv",
                          "data/y_coords.csv",
                          file,
                          f"results/{name}_{i}.csv")