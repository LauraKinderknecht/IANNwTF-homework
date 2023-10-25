import numpy as np

#Create a 5x5 NumPy array filled with normally distributed values (i.e. μ = 0, σ = 1)
array = np.random.normal(loc=0.0, scale=1.0, size=(5,5))

#initial array
print("Initial 5x5 NumPy array:")
print(array)
print()

#go through every row in the array
for row_idx, row in enumerate(array):
    #for each row, go through every column/element
    for col_idx, elem in enumerate(row):
        #If the value of an entry is greater than 0.09, replace it with its square. 
        if elem > 0.09:
            array[row_idx][col_idx] = round(elem**2,3)
        #Else, replace it with 42.
        else:
            array[row_idx][col_idx] = int(42)

print("After step 2:")        
print(array)
print()

#take all rows and only take the fourth column - using slicing
print("After step 3: slicing")
sliced_array = array[::,3::2]
print(sliced_array)

