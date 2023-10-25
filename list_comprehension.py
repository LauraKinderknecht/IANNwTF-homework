#list of the squares of each number between 0 to 100
numbers = [num**2 for num in range(0,101)]

print("List of the squares of each number between 0 to 100:")
print(numbers)
print()

#list of the squares of the numbers between 0 to 100 - which are even squares
even = [num**2 for num in range(0,101) if (num**2)%2==0]

print("List of the squares of the numbers between 0 to 100 - which are even squares:")
print(even)
