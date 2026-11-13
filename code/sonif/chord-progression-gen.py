from itertools import product

states = ("I", "II", "IV", "V", "VI", "VII")

firstChord = "I"
cadence = ("V", "I")

numChordEvents = 11
numFreeChords = numChordEvents - 1 - len(cadence)
count = 0

for middleChords in product(states, repeat=numFreeChords):
    progression = (firstChord,) + middleChords + cadence 

    # Calculate E_geometry and E_harmony.
    # Calculate E_total.
    # Retain the best progression found.

    count += 1

print("Total progressions:", count)
