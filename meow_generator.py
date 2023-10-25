def generate_meows():
    #start with one "meow" for the first call
    meow = 1

    #yields one "Meow" for the first call 
    #and then twice the amount of "Meows" for each successive call    
    while True:
        #yields the string "Meow " meow-times
        yield "Meow " * meow
        #doubles the number of "meows" after each function call
        meow *= 2


gen_meows = generate_meows()

#first function call
#output: Meow
print(next(gen_meows))  

#second call
#output: Meow Meow
print(next(gen_meows))

#third call
#output: Meow Meow Meow Meow
print(next(gen_meows))

#fourth call
#output: Meow Meow Meow Meow Meow Meow Meow Meow
print(next(gen_meows))
