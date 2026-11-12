import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import numpy as np

DATA_DIR   = PRJ_DIR / "data" / "greenness"
OUTPUT_DIR = PRJ_DIR / "data" / "sonif-input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imageDates   = ("20240801", "20250201")
hemispheres  = ("north", "south")
regionWidths = (4, 2)

numRows = 16
numCols = 32

# Area represented by an equirectangular cell is proportional to
# cos(latitude). Rows run from 90 N to 90 S.
latCenters = 90.0 - (np.arange(numRows) + 0.5) * (180.0 / numRows)
latWeights = np.cos(np.deg2rad(latCenters))


def loadGridData(imageDate):
    inputNpz = f"{imageDate}-griddata-d35.npz"
    with np.load(DATA_DIR / inputNpz) as data:
        requiredKeys = ("meanScores", "stdScores", "validFractions", "snowFractions", "landFractions")
        for key in requiredKeys:
            if key not in data: raise KeyError(f"Missing {key} in {inputNpz}")

        gridData = {key: data[key] for key in requiredKeys}

    meanScores     = gridData["meanScores"]
    stdScores      = gridData["stdScores"]
    validFractions = gridData["validFractions"]
    snowFractions  = gridData["snowFractions"]
    landFractions  = gridData["landFractions"]

    # Validate input array shapes.
    for name, values in gridData.items():
        if values.shape != (numRows, numCols):
            raise ValueError( f"{name}'s shape: {values.shape}; expected ({numRows}, {numCols})" )

    # Validate coverage fractions.
    assert np.all(np.isfinite(validFractions))
    assert np.all(np.isfinite(snowFractions))
    assert np.all(np.isfinite(landFractions))

    assert np.all((validFractions >= 0) & (validFractions <= 1))
    assert np.all((snowFractions >= 0) & (snowFractions <= 1))
    assert np.all((landFractions >= 0) & (landFractions <= 1 + 1e-10))

    assert np.all(validFractions + snowFractions <= 1 + 1e-10)
    assert np.allclose( landFractions, validFractions + snowFractions, atol=1e-10 )

    # Validate greenness statistics.
    hasValid = validFractions > 0

    assert np.all(np.isnan(meanScores[~hasValid]))
    assert np.all(np.isnan(stdScores[~hasValid]))
    assert np.all(np.isfinite(meanScores[hasValid]))
    assert np.all(np.isfinite(stdScores[hasValid]))

    assert np.all((meanScores[hasValid] >= 0) & (meanScores[hasValid] <= 1))
    assert np.all((stdScores[hasValid] >= 0) & (stdScores[hasValid] <= 0.5 + 1e-10))

    return gridData


def aggregateRegions(imageDate, hemisphere, colsPerRegion, gridData):
    if hemisphere == "north":
        rowStart = 0
        rowEnd = numRows // 2
    elif hemisphere == "south":
        rowStart = numRows // 2
        rowEnd = numRows
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")

    if colsPerRegion <= 0 or numCols % colsPerRegion != 0:
        raise ValueError(f"Invalid colsPerRegion: {colsPerRegion}")

    numRegions = numCols // colsPerRegion

    meanScores     = gridData["meanScores"]
    stdScores      = gridData["stdScores"]
    validFractions = gridData["validFractions"]
    snowFractions  = gridData["snowFractions"]
    landFractions  = gridData["landFractions"]

    # Initialize regional statistics.
    regionMeanScores       = np.full(numRegions, np.nan)
    regionStdScores        = np.full(numRegions, np.nan)
    regionLandGreenness    = np.full(numRegions, np.nan)
    regionLandGreennessStd = np.full(numRegions, np.nan)

    regionValidFractions = np.zeros(numRegions)
    regionSnowFractions  = np.zeros(numRegions)
    regionLandFractions  = np.zeros(numRegions)

    # Latitude-area weights for the selected hemisphere.
    areaWeights = np.broadcast_to( latWeights[rowStart:rowEnd, np.newaxis],
                                   (rowEnd - rowStart, colsPerRegion))
    totalAreaWeight = np.sum(areaWeights)

    for region in range(numRegions):
        colStart = region * colsPerRegion
        colEnd = colStart + colsPerRegion

        regionSlice = ( slice(rowStart, rowEnd), slice(colStart, colEnd) )

        means = meanScores[regionSlice]
        stds  = stdScores[regionSlice]
        valid = validFractions[regionSlice]
        snow  = snowFractions[regionSlice]
        land  = landFractions[regionSlice]

        # All three coverage fractions use the entire mapped
        # geographical region as their denominator.
        regionValidFractions[region] = np.sum(areaWeights * valid) / totalAreaWeight
        regionSnowFractions[region]  = np.sum(areaWeights * snow) / totalAreaWeight
        regionLandFractions[region]  = np.sum(areaWeights * land) / totalAreaWeight

        # Identify cells containing valid greenness measurements.
        usable = ( (valid > 0) & np.isfinite(means) & np.isfinite(stds) )

        # Weight valid greenness by latitude and accepted coverage.
        greennessWeights = areaWeights * valid

        # Land-adjusted greenness and standard deviation.
        # Classified snow/ice is treated as zero greenness;
        # ocean and unmatched pixels are excluded.
        totalLandWeight = np.sum(areaWeights * land)

        if totalLandWeight > 0:
            totalGreenness = np.sum( greennessWeights[usable] * means[usable] )

            regionLandMean = totalGreenness / totalLandWeight
            regionLandGreenness[region] = regionLandMean

            # Second moment of land-adjusted greenness,
            # including within-cell greenness variation.
            totalSecondMoment = np.sum( greennessWeights[usable] * (stds[usable]**2 + means[usable]**2) )

            regionLandVariance = ( totalSecondMoment / totalLandWeight - regionLandMean**2 )

            regionLandGreennessStd[region] = np.sqrt( max(regionLandVariance, 0.0) )

        # Regional greenness statistics of valid pixels only.
        if not np.any(usable):
            continue

        weights = greennessWeights[usable]
        cellMeans = means[usable]
        cellStds = stds[usable]

        # Regional mean greenness.
        regionMean = np.average(cellMeans, weights=weights)
        regionMeanScores[region] = regionMean

        # Regional pooled variance:
        # within-cell + between-cell greenness variation.
        regionVariance = np.average( cellStds**2 + (cellMeans - regionMean)**2, weights=weights )

        regionStdScores[region] = np.sqrt( max(regionVariance, 0.0) )

    # Validate aggregated fractions.
    assert np.allclose( regionLandFractions, regionValidFractions + regionSnowFractions )

    # Validate the relationship between land-adjusted and
    # valid-pixel mean greenness.
    usableRegions = ( (regionLandFractions > 0) & (regionValidFractions > 0) )

    assert np.allclose(
        regionLandGreenness[usableRegions],
        regionMeanScores[usableRegions] 
        * regionValidFractions[usableRegions]
        / regionLandFractions[usableRegions] )

    for values in (regionMeanScores, regionStdScores, regionLandGreenness, regionLandGreennessStd):
        finiteValues = values[np.isfinite(values)]
        assert np.all(finiteValues >= 0)
        assert np.all(finiteValues <= 1 + 1e-10)

    assert np.all( regionStdScores[np.isfinite(regionStdScores)] <= 0.5 + 1e-10 )
    assert np.all( regionLandGreennessStd[np.isfinite(regionLandGreennessStd)] <= 0.5 + 1e-10 )

    # Region numbers and longitude boundaries.
    regionNumbers = np.arange(1, numRegions + 1)
    longitudeWidth = 360.0 / numRegions

    longitudeStarts = -180.0 + np.arange(numRegions) * longitudeWidth
    longitudeEnds = longitudeStarts + longitudeWidth

    outputNpz = f"{imageDate}-{hemisphere}-regions-{colsPerRegion}col.npz"
    outputCsv = f"{imageDate}-{hemisphere}-regions-{colsPerRegion}col.csv"

    # Save regional statistics in NPZ format.
    np.savez(
        OUTPUT_DIR / outputNpz,
        regionNumbers=regionNumbers,
        longitudeStarts=longitudeStarts,
        longitudeEnds=longitudeEnds,
        regionMeanScores=regionMeanScores,
        regionStdScores=regionStdScores,
        regionLandGreenness=regionLandGreenness,
        regionLandGreennessStd=regionLandGreennessStd,
        regionValidFractions=regionValidFractions,
        regionSnowFractions=regionSnowFractions,
        regionLandFractions=regionLandFractions)
    print(outputNpz, "saved.")

    # Combine regional statistics into a CSV table.
    outputData = np.column_stack([
        regionNumbers,
        longitudeStarts,
        longitudeEnds,
        regionMeanScores,
        regionStdScores,
        regionLandGreenness,
        regionLandGreennessStd,
        regionValidFractions,
        regionSnowFractions,
        regionLandFractions ])

    np.savetxt(
        OUTPUT_DIR / outputCsv,
        outputData,
        delimiter=",",
        header=(
            "region,longitudeStart,longitudeEnd,"
            "meanScore,stdScore,"
            "landGreenness,landGreennessStd,"
            "validFraction,snowFraction,landFraction" ),
        comments="",
        fmt=["%d", "%.1f", "%.1f"] + ["%.6f"] * 7 )
    print(outputCsv, "saved.")


# Generate all hemisphere/resolution combinations for each date.
for imageDate in imageDates:
    print(f"\nProcessing {imageDate}")

    gridData = loadGridData(imageDate)

    for hemisphere in hemispheres:
        for colsPerRegion in regionWidths:
            aggregateRegions( imageDate, hemisphere, colsPerRegion, gridData)
