import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import csv, math
from constants import DATA_SONIF_DIR, IMAGE_DATES

HEMISPHERE = "north"

oceanThreshold = 0.10
landThreshold = 0.25
lowGreennessThreshold = 0.25
highGreennessThreshold = 0.55

numCols = 32
coarseCols = 4
fineCols = 2
beatsPerColumn = 0.5
totalBeats = numCols * beatsPerColumn

featureNames = ("meanScore",
                "stdScore",
                "landGreenness",
                "landGreennessStd",
                "validFraction",
                "snowFraction",
                "landFraction")

outputFields = ("eventIndex",
                "parentRegion",
                "chordEventIndex",
                "firstColumn",
                "lastColumn",
                "longitudeStart",
                "longitudeEnd",
                "startBeat",
                "durationBeats") + featureNames


# Load regional environmental statistics.
#
def loadRegionCsv(imageDate, colsPerRegion):
    inputCsv = f"{imageDate}-{HEMISPHERE}-regions-{colsPerRegion}col.csv"

    with open(DATA_SONIF_DIR / inputCsv, newline="") as file:
        rows = list(csv.DictReader(file))

    expectedRegions = numCols // colsPerRegion

    if len(rows) != expectedRegions:
        raise ValueError(
            f"{inputCsv}: expected {expectedRegions} regions, found {len(rows)}"
        )

    for index, row in enumerate(rows):
        if int(row["region"]) != index + 1:
            raise ValueError(f"Invalid region number in {inputCsv}")

        expectedStart = -180.0 + index * colsPerRegion * 11.25
        expectedEnd = expectedStart + colsPerRegion * 11.25

        # Accept both one- and two-decimal longitude CSV formats.
        if not math.isclose(
            float(row["longitudeStart"]), expectedStart, abs_tol=0.06
        ):
            raise ValueError(f"Invalid longitudeStart in {inputCsv}")

        if not math.isclose(
            float(row["longitudeEnd"]), expectedEnd, abs_tol=0.06
        ):
            raise ValueError(f"Invalid longitudeEnd in {inputCsv}")

    return rows


# Load Southern Hemisphere chord-event timing.
# The CSV specifies when chords change, not their Roman numerals.
#
def loadChordEvents(imageDate):
    inputCsv = f"{imageDate}-south-regions-chords.csv"

    with open(DATA_SONIF_DIR / inputCsv, newline="") as file:
        rows = list(csv.DictReader(file))

    chordEvents = []
    expectedStart = 0.0

    for row in rows:
        start = float(row["startBeat"])
        duration = float(row["durationBeats"])
        end = start + duration

        if duration <= 0 or not math.isclose(start, expectedStart):
            raise ValueError(f"Invalid chord timing in {inputCsv}")

        chordEvents.append((start, end))
        expectedStart = end

    if not math.isclose(expectedStart, totalBeats):
        raise ValueError(f"Chord events do not fill {totalBeats} beats.")

    return chordEvents


# Return greenness, using zero only as a fallback for a region
# containing no classified land.
#
def getGreenness(row):
    greenness = float(row["landGreenness"])

    if math.isfinite(greenness):
        return greenness

    if float(row["landFraction"]) == 0:
        return 0.0

    raise ValueError("Nonfinite landGreenness in a region containing land.")


# Estimate the desirability of subdividing one beat into
# two eighth notes.
#
# 2-column statistics describe the beat.
# 1-column statistics capture local variation within it.
#
def calculateBeatActivity(fineRow, onePair):
    land = float(fineRow["landFraction"])
    greenness = getGreenness(fineRow)

    landValues = [float(row["landFraction"]) for row in onePair]
    greenValues = [getGreenness(row) for row in onePair]

    landContrast = abs(landValues[1] - landValues[0])

    # Avoid giving large influence to greenness measured
    # over very small land areas.
    greenContrast = (
        min(landValues) * abs(greenValues[1] - greenValues[0])
    )

    return (
        land * (0.5 + 0.5 * greenness)
        + 0.25 * landContrast
        + 0.25 * greenContrast
    )


# Select the rhythmic pattern for one 4-column region.
# Each pattern occupies exactly two beats.
#
def selectRhythm(coarseRow, finePair, oneQuad):
    land = float(coarseRow["landFraction"])

    # Ocean-dominated: one sustained half note.
    if land < oceanThreshold:
        return [2.0]

    # Sparse land: two quarter notes.
    if land < landThreshold:
        return [1.0, 1.0]

    greenness = getGreenness(coarseRow)

    # Substantial land with low greenness: calm melody.
    if greenness < lowGreennessThreshold:
        return [1.0, 1.0]

    # Substantial land with high greenness: lively melody.
    if greenness > highGreennessThreshold:
        return [0.5, 0.5, 0.5, 0.5]

    # Moderate greenness: one quarter and two eighths.
    # Place the eighth notes in the more active beat.
    firstBeatActivity = calculateBeatActivity(
        finePair[0], oneQuad[:2]
    )
    secondBeatActivity = calculateBeatActivity(
        finePair[1], oneQuad[2:]
    )

    if firstBeatActivity >= secondBeatActivity:
        return [0.5, 0.5, 1.0]
    else:
        return [1.0, 0.5, 0.5]


# Combine statistics from consecutive 1-column regions.
# Fraction means are area averages.
# Greenness means and standard deviations are pooled using
# the appropriate valid or land coverage weights.
#
def aggregateEventFeatures(rows):

    def pooledStatistics(fractionName, meanName, stdName):
        values = []

        for row in rows:
            weight = float(row[fractionName])

            if weight > 0:
                mean = float(row[meanName])
                std = float(row[stdName])

                if not math.isfinite(mean) or not math.isfinite(std):
                    raise ValueError("Invalid regional statistics.")

                values.append((weight, mean, std))

        if not values:
            return float("nan"), float("nan")

        totalWeight = sum(weight for weight, _, _ in values)

        pooledMean = sum(
            weight * mean for weight, mean, _ in values
        ) / totalWeight

        secondMoment = sum(
            weight * (std**2 + mean**2)
            for weight, mean, std in values
        ) / totalWeight

        pooledStd = math.sqrt(
            max(secondMoment - pooledMean**2, 0.0)
        )

        return pooledMean, pooledStd

    features = {}

    for name in ("validFraction", "snowFraction", "landFraction"):
        features[name] = sum(
            float(row[name]) for row in rows
        ) / len(rows)

    features["meanScore"], features["stdScore"] = pooledStatistics(
        "validFraction", "meanScore", "stdScore"
    )

    features["landGreenness"], features["landGreennessStd"] = (
        pooledStatistics(
            "landFraction", "landGreenness", "landGreennessStd"
        )
    )

    return features


# Generate rhythmic events, splitting notes at chord changes.
#
def generateMelodyRhythm(coarseRows, fineRows, oneRows, chordEvents):
    melodyEvents = []

    chordBoundaries = [
        start for start, _ in chordEvents[1:]
    ]

    for parentIndex, coarseRow in enumerate(coarseRows):
        fineIndex = parentIndex * 2
        oneIndex = parentIndex * 4

        finePair = fineRows[fineIndex:fineIndex + 2]
        oneQuad = oneRows[oneIndex:oneIndex + 4]

        durations = selectRhythm(
            coarseRow, finePair, oneQuad
        )

        assert math.isclose(sum(durations), 2.0)

        currentBeat = parentIndex * 2.0

        for duration in durations:
            noteEnd = currentBeat + duration

            # Split any note that crosses one or more chord changes.
            splitPoints = (
                [currentBeat]
                + [b for b in chordBoundaries if currentBeat < b < noteEnd]
                + [noteEnd]
            )

            for startBeat, endBeat in zip(
                splitPoints[:-1], splitPoints[1:]
            ):
                firstColumn = int(round(startBeat / beatsPerColumn)) + 1
                lastColumn = int(round(endBeat / beatsPerColumn))

                coveredRows = oneRows[firstColumn - 1:lastColumn]

                chordEventIndex = next(
                    index + 1
                    for index, (chordStart, chordEnd) in enumerate(chordEvents)
                    if chordStart <= startBeat and endBeat <= chordEnd
                )

                event = {
                    "eventIndex": len(melodyEvents) + 1,
                    "parentRegion": parentIndex + 1,
                    "chordEventIndex": chordEventIndex,
                    "firstColumn": firstColumn,
                    "lastColumn": lastColumn,
                    "longitudeStart": -180.0 + (firstColumn - 1) * 11.25,
                    "longitudeEnd": -180.0 + lastColumn * 11.25,
                    "startBeat": startBeat,
                    "durationBeats": endBeat - startBeat
                }

                event.update(aggregateEventFeatures(coveredRows))
                melodyEvents.append(event)

            currentBeat = noteEnd

    # Verify temporal continuity and total duration.
    expectedStart = 0.0

    for event in melodyEvents:
        assert math.isclose(event["startBeat"], expectedStart)
        expectedStart += event["durationBeats"]

    assert math.isclose(expectedStart, totalBeats)

    return melodyEvents


# Save the melody rhythm and environmental features.
#
def saveMelodyRhythm(imageDate, melodyEvents):
    outputCsv = f"{imageDate}-{hemisphere}-regions-melody-rhythm.csv"

    with open(DATA_SONIF_DIR / outputCsv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=outputFields)
        writer.writeheader()
        writer.writerows(melodyEvents)

    print(outputCsv, "saved.")
    print("Total melody events:", len(melodyEvents))
    print("Total beats:", sum(
        event["durationBeats"] for event in melodyEvents
    ))

    print("Melody rhythm:")
    for event in melodyEvents:
        print(
            f"  event={event['eventIndex']:>2} "
            f"chord={event['chordEventIndex']:>2} "
            f"start={event['startBeat']:>4.1f} "
            f"duration={event['durationBeats']:>3.1f} "
            f"land={event['landFraction']:.3f} "
            f"greenness={event['landGreenness']:.3f}"
        )


# Process each image date.
#
for imageDate in IMAGE_DATES:
    print(f"\nProcessing {imageDate}")

    coarseRows = loadRegionCsv(imageDate, coarseCols)
    fineRows = loadRegionCsv(imageDate, fineCols)
    oneRows = loadRegionCsv(imageDate, 1)

    chordEvents = loadChordEvents(imageDate)

    melodyEvents = generateMelodyRhythm(
        coarseRows, fineRows, oneRows, chordEvents
    )

    saveMelodyRhythm(imageDate, melodyEvents)
