class Cat:

    #constructor that creates a new instances of the "Cat" class
    def __init__(self, name = "Max"):
        #cat's name is set as the name in the user input
        #or if no name was in the user input, then the cat's name is set to "Max" by default
        self.name = name
        
    #greet method prints 
    def greet(self,other_cat):
        print(f"Hi. I'm {self.name}! What is up {other_cat.name}, my feline friend? Wanna chase some birds?")
    
