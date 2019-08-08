import numpy as np
import warnings


def determineThreshold(array, maxSamples=50000):
    array = np.array(array)
    elements = len(array)

    if elements > maxSamples:  # subsample
        step = round(elements / maxSamples)
        array = array[0::step]
        elements = len(array)

    connectingline = np.linspace(array[0], array[-1], elements)
    distances = np.abs(array - connectingline)
    position = np.argmax(distances)

    threshold = array[position]
    if np.isnan(threshold):
        threshold = 0
    return threshold


def selectiveMedianFilter(
    stack,
    backgroundValue=0,
    medianRange=3,
    verbose=False,
    withMean=False,
    deviationThreshold=None,
):
    """correct bad pixels on sCMOS camera.
    based on MATLAB code by Philipp J. Keller,
    HHMI/Janelia Research Campus, 2011-2014

    """
    from scipy.ndimage.filters import median_filter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        devProj = np.std(stack, 0, ddof=1)
        devProjMedFiltered = median_filter(devProj, medianRange, mode="constant")
        deviationDistances = np.abs(devProj - devProjMedFiltered)
        deviationDistances[deviationDistances == np.inf] = 0
        if deviationThreshold is None:
            deviationThreshold = determineThreshold(
                sorted(deviationDistances.flatten())
            )

        deviationMatrix = deviationDistances > deviationThreshold

        if withMean:
            meanProj = np.mean(stack, 0) - backgroundValue
            meanProjMedFiltered = median_filter(meanProj, medianRange)
            meanDistances = np.abs(meanProj - meanProjMedFiltered / meanProjMedFiltered)
            meanDistances[meanDistances == np.inf] = 0
            meanThreshold = determineThreshold(sorted(meanDistances.flatten()))

            meanMatrix = meanDistances > meanThreshold

            pixelMatrix = deviationMatrix | meanMatrix
            pixelCorrection = [
                deviationDistances,
                deviationThreshold,
                meanDistances,
                meanThreshold,
            ]
        else:
            pixelMatrix = deviationMatrix
            pixelCorrection = [deviationDistances, deviationThreshold]

        if verbose:
            pixpercent = (
                100 * np.sum(pixelMatrix.flatten()) / float(len(pixelMatrix.flatten()))
            )
            print(
                "Bad pixels detected: {} {:0.2f}".format(
                    np.sum(pixelMatrix.flatten()), pixpercent
                )
            )

        dt = stack.dtype
        out = np.zeros(stack.shape, dt)
        # apply pixelMatrix to correct insensitive pixels
        for z in range(stack.shape[0]):
            frame = np.asarray(stack[z], "Float32")
            filteredFrame = median_filter(frame, medianRange)
            frame[pixelMatrix == 1] = filteredFrame[pixelMatrix == 1]
            out[z] = np.asarray(frame, dt)

        return out, pixelCorrection
