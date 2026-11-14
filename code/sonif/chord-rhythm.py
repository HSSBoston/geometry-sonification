import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import csv, math
from constants import SONIF_INPUT_DIR, SONIF_DIR

imageDates = ("20240501", "20240801", "20241101", "20250201")

hemisphere = "south"
landThreshold = 0.25

numCols = 32
coarseCols = 4
fineCols = 2
numCoarseRegions = numCols // coarseCols

featureNames = ("meanScore",
                "stdScore",
                "landGreenness",
                "landGreennessStd",
                "validFraction",
                "snowFraction",
                "landFraction")

outputFields = ("eventIndex",
                "regionLabel",
                "parentRegion",
                "subRegion",
                "colsPerRegion",
                "longitudeStart",
                "longitudeEnd",
                "startBeat",
                "durationBeats") + featureNames


def loadRegionCsv(imageDate, colsPerRegion):
    inputCsv = f"{imageDate}-{hemisphere}-regions-{colsPerRegion}col.csv"

    with open(SONIF_INPUT_DIR / inputCsv, newline="") as file:
        reader = csv.DictReader(file)

        requiredFields = ("region", "longitudeStart", "longitudeEnd") + featureNames

        missingFields = [ field for field in requiredFields
                          if field not in reader.fieldnames ]
        if missingFields:
            raise ValueError(f"Missing columns in {inputCsv}: {missingFields}")

        rows = list(reader)

    expectedRegions = numCols // colsPerRegion
    if len(rows) != expectedRegions:
        raise ValueError( f"{inputCsv}: expected {expectedRegions} rows, found {len(rows)}" )

    # Verify region numbering and longitude boundaries.
    longitudeWidth = 360.0 / expectedRegions

    for index, row in enumerate(rows):
        expectedRegion = index + 1
        expectedStart = -180.0 + index * longitudeWidth
        expectedEnd = expectedStart + longitudeWidth

        if int(row["region"]) != expectedRegion:
            raise ValueError( f"Unexpected region number in {inputCsv}" )

        if not math.isclose( float(row["longitudeStart"]), expectedStart, abs_tol=1e-6):
            raise ValueError( f"Invalid longitudeStart for region {expectedRegion}" )

        if not math.isclose( float(row["longitudeEnd"]), expectedEnd, abs_tol=1e-6 ):
            raise ValueError( f"Invalid longitudeEnd for region {expectedRegion}" )

        landFraction = float(row["landFraction"])

        if not math.isfinite(landFraction) or not (0 <= landFraction <= 1):
            raise ValueError(f"Invalid landFraction for region {expectedRegion}")
    return rows


def generateAdaptiveRegions(coarseRows, fineRows):
    adaptiveRows = []
    currentBeat = 0

    for parentIndex, coarseRow in enumerate(coarseRows):
        parentRegion = parentIndex + 1
        landFraction = float(coarseRow["landFraction"])

        # The two fine regions corresponding to this coarse region.
        fineIndex = parentIndex * 2
        finePair = fineRows[fineIndex:fineIndex + 2]

        # Check consistency of land fractions across resolutions.
        expectedLandFraction = (float(finePair[0]["landFraction"]) + float(finePair[1]["landFraction"]))/2.0
        if not math.isclose( landFraction, expectedLandFraction, abs_tol=2e-6):
            raise ValueError(f"Inconsistent landFraction for region {parentRegion}")

        # Decide whether to subdivide the region.
        if landFraction >= landThreshold:
            selectedRegions = [ (finePair[0], f"{parentRegion}a", 1, fineCols, 1),
                                (finePair[1], f"{parentRegion}b", 2, fineCols, 1) ]
        else:
            selectedRegions = [ (coarseRow, str(parentRegion), 0, coarseCols, 2) ]

        # Add one or two chord events.
        for row, label, subRegion, width, duration in selectedRegions:
            event = { "eventIndex":    len(adaptiveRows) + 1,
                      "regionLabel":   label,
                      "parentRegion":  parentRegion,
                      "subRegion":     subRegion,
                      "colsPerRegion": width,
                      "longitudeStart":row["longitudeStart"],
                      "longitudeEnd":  row["longitudeEnd"],
                      "startBeat":     currentBeat,
                      "durationBeats": duration }

            # Retain the original environmental statistics.
            for feature in featureNames:
                event[feature] = row[feature]

            adaptiveRows.append(event)
            currentBeat += duration

    # Eight coarse regions, each occupying two beats.
    expectedBeats = numCoarseRegions * 2
    assert currentBeat == expectedBeats

    # Check geographical continuity.
    for index in range(1, len(adaptiveRows)):
        previousEnd  = float( adaptiveRows[index - 1]["longitudeEnd"] )
        currentStart = float( adaptiveRows[index]["longitudeStart"] )
        assert math.isclose( previousEnd, currentStart, abs_tol=1e-6 )
    return adaptiveRows


def saveAdaptiveRegions(imageDate, adaptiveRows):
    outputCsv = f"{imageDate}-{hemisphere}-chords.csv"

    with open(SONIF_DIR / outputCsv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=outputFields)
        writer.writeheader()
        writer.writerows(adaptiveRows)

    print(outputCsv, "saved.")
    print(f"Total chord events: {len(adaptiveRows)}")
    print("Total beats:", sum(int(row["durationBeats"]) for row in adaptiveRows) )

    print("Chord sequence:")
    for row in adaptiveRows:
        print( f"  {row['regionLabel']:>3}  "
               f"start={row['startBeat']:>2}  "
               f"duration={row['durationBeats']}  "
               f"land={float(row['landFraction']):.4f}" )

for imageDate in imageDates:
    print(f"\nProcessing {imageDate}")
    coarseRows = loadRegionCsv(imageDate, coarseCols)
    fineRows   = loadRegionCsv(imageDate, fineCols)

    adaptiveRows = generateAdaptiveRegions(coarseRows, fineRows)
    saveAdaptiveRegions(imageDate, adaptiveRows)
