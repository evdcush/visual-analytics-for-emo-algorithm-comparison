import numpy as np
from sampling.Sampler import *
from sampling.SamplingMethods import *
from sklearn.neighbors import LocalOutlierFactor


def sampler(algoData, sampling_rate):
    all_solutions_data = []
    all_solutions_class = []
    outlier = np.array([])
    for index, (_, data) in enumerate(algoData.items()):
        outlier = np.concatenate((outlier, LocalOutlierFactor().fit_predict(data)))
        all_solutions_data += data
        all_solutions_class += [index] * len(data)
    outlier_indices = np.where(outlier == -1)[0]
    indices = np.zeros(len(all_solutions_data))
    if sampling_rate == 0:
        return indices
    if sampling_rate == 1:
        indices = np.ones(len(all_solutions_data))
        indices[outlier_indices] = -1
        return indices
    sampler = Sampler()
    sampler.set_data(np.array(all_solutions_data),
                     np.array(all_solutions_class))
    sampling_method = OutlierBiasedDensityBasedSampling
    args = {  # This sampling method do not need sampling rate as input
        'sampling_rate': sampling_rate,
    }
    sampler.set_sampling_method(sampling_method, **args)
    selected_indices = sampler.get_samples_idx()
    indices[selected_indices] = 1
    indices[outlier_indices] = -1
    return indices
