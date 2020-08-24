class demoRFF:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def greet(self):
        print(f'Hello {self.name}, aged {self.age} years')