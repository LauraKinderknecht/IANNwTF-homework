def generate_meows(n=20):
    string = "meow"
    while n>21:
        yield string
        string += string
        n += 1

meows = generate_meows()

print(meows)