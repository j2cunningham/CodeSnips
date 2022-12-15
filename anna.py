#Task 1: make a dictionary
rarebirds = {
   'Gold-crested Toucan': {
        'Height (m)': '1.1',
        'Weight (kg)': '35',
        'Color': 'Gold',
        'Endangered': True,
        'Aggressive': True},
    'Pearlescent Kingfisher': {
        'Height (m)': '0.25',
        'Weight (kg)': '0.5',
        'Color': 'White',
        'Endangered': False,
        'Aggressive': False}, 
    'Four-metre Hummingbird': {
        'Height (m)': '0.6',
        'Weight (kg)': '0.5',
        'Color': 'Blue',
        'Endangered': True,
        'Aggressive': False},
    'Giant Eagle': {
        'Height (m)': '1.5',
        'Weight (kg)': '52',
        'Color': 'Black and White',
        'Endangered': True,
        'Aggressive': True},
    'Ancient Vulture': {
        'Height (m)': '2.1',
        'Weight (kg)': '70',
        'Color': 'Blown',
        'Endangered': False,
        'Aggressive': False}
    }

#Task 2: make a list for locations
birdlocation = ['in the canopy directly above our heads', 
    'between my 6 and 9 o’clock above', 
    'between my 9 and 12 o’clock above', 
    'between my 12 and 3 o’clock above', 
    'between my 3 and 6 o’clock above', 
    'in a nest on the ground', 
    'right behind you']

#Task 3: create binary codes
codes = {'111':'in the canopy directly above our heads.',
        '110':'between my 6 and 9 o’clock above', 
        '101':'between my 9 and 12 o’clock above',
        '100':'between my 12 and 3 o’clock above',
        '011':'between my 3 and 6 o’clock above',
        '010':'in a nest on the ground',
        '001':'right behind you'}

#Task 4: create list of actions
actions = ['back away', 'cover our heads', 'take a photograph']

#Task 5: check if the giant eagle is aggressive
print(rarebirds['Giant Eagle']['Aggressive'])

#Task 6: create a for loop that goes through keys and values of rarebirds dictionary
#Check what birds are aggressive
for key, value in rarebirds.items():
    if value['Aggressive'] == True:
        print(key + ": " + actions[1])

#Check what birds are endangered
for key, value in rarebirds.items():
    if value['Endangered'] == True:
        print(key + ": " + actions[0])

#Task 7: create a for loop to describe each code
for key, value in codes.items():
    print(key + ": " + value)

#Task 8: create a for loop to add an extra attribute, 'seen'
rarebirds['Gold-crested Toucan']['Seen'] = False
rarebirds['Pearlescent Kingfisher']['Seen'] = False
rarebirds['Four-metre Hummingbird']['Seen'] = False
rarebirds['Giant Eagle']['Seen'] = False
rarebirds['Ancient Vulture']['Seen'] = False

#Task 9: new variable
encounter = True

#Task 17: Using your encounter variable within a big while-loop, modify the above code so that input is repeatedly requested 
# from the user until they input one of the birds in our rarebirds dictionary. Hint: start the encounter by making encounter’s value True. 
# Make encounter’s value False within the nested if, elif and else statements.
while encounter == True:
    #Task 10: ask for input
    sighting = input('What do you see? ')
    #Task 11: create a list of rare birds keyes
    rarebirdsList = list(rarebirds.keys())
    #This was a test
    #print(rarebirdsList)
    #Task 12: check if sighting (input) is in the list
    if sighting in rarebirdsList:  
        print("This is one of the birds we're looking for!")
        encounter = False
    else:
        print("That's not one of the birds we're looking for.")
    #Task 13: ask for more input; store as code
    code = input('Where do you see it? Input the correct code. ')
    #Task 14: new variable for location
    location = codes[code]
    #Task 15: print out a statement using sighting and location
    print("So you've seen a " + sighting + " " + location + " My goodness!")
    #Task 16: Let's do some more logic. Make an if, elif, and else statement. In the if-statement, check whether the sighting is aggressive. 
    # If it is, print out that it’s aggressive, and that we need to back away and cover our heads. In addition, print out that we need to photograph 
    # the sighting at its location. In the elif-statement, check whether the sighting is endangered. If it is, print out that it’s endangered, 
    # and that we need to back away. Also, print out that we need to photograph the sighting at its location. In the else statement (i.e, the sighting is 
    # neither aggressive nor endangered) print out that we need to photograph the ultra rare sighting at its location. In all of these blocks, make use of 
    # the actions list and the variables sighting and location in your printouts.
    # for bird in rarebirds:
    #     trait = rarebirds[bird]['Aggressive']
    #     if sighting == True:
    #         print("It's aggressive, we need to " + actions[0] + "and" + actions[1])
    #         print("We need to photograph the " + sighting + " " + location + ".")
    #         encounter = False
    #     elif trait == 'Endangered':
    #         print("It's endangered, so we need to " + actions[0])
    #         print("We need to photograph the " + sighting + " " + location + ".")
    #         encounter = False
    #     else:
    #         print("We need to photograph this ultra rare " + sighting + " " + location + ".")
    #         encounter = False

    if encounter == False:
        trait = rarebirds[sighting]['Aggressive']
        if trait:
            print("It's aggressive, we need to " + actions[0] + " and " + actions[1])
            print("We need to photograph the " + sighting + " " + location + ".")
        elif trait == 'Endangered':
            print("It's endangered, so we need to " + actions[0])
            print("We need to photograph the " + sighting + " " + location + ".")
        else:
            print("We need to photograph this ultra rare " + sighting + " " + location + ".")

    
