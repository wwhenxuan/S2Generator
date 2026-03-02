import numpy as np
from numpy import fft


def sample_random_perturbation(
    K: int, min_alpha: float, max_alpha: float, rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Randomly sample K numbers in the interval [-alpha_max, -alpha_min] U [alpha_min, alpha_max]
    The purpose of this sampling is to construct random perturbations in the frequency domain.

    :param K: Number of random numbers to sample
    :param min_alpha: Minimum absolute value of the random numbers
    :param max_alpha: Maximum absolute value of the random numbers
    :param rng: Optional random number generator, if not provided, the global numpy random number generator will be used

    :return: A numpy array containing K random numbers, which are uniformly distributed in the interval [-alpha_max, -alpha_min] U [alpha_min, alpha_max]
    """

    # First generate random numbers in [alpha_min, alpha_max]
    if rng is not None:
        positive_rand = rng.uniform(min_alpha, max_alpha, K)

        # Randomly generate sign (-1 or 1)
        signs = rng.choice([-1, 1], size=K)

    else:
        # When the random number generator is not passed in, use the global numpy random number generator
        positive_rand = np.random.uniform(min_alpha, max_alpha, K)
        signs = np.random.choice([-1, 1], size=K)

    # Combine to get the final result
    final_random_nums = positive_rand * signs

    return final_random_nums


def frequency_perturbation(
    series: np.ndarray,
    min_alpha: float,
    max_alpha: float,
    r: float = 0.5,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """
    Perform frequency domain perturbation on the input time series.
    This method adds random perturbations to the frequency components of the time series,
    which can help to enhance the diversity of the data and improve the robustness of models trained on it.

    :param series: Input time series, a 1D numpy array
    :param min_alpha: Minimum absolute value of the random perturbation added to the frequency components
    :param max_alpha: Maximum absolute value of the random perturbation added to the frequency components
    :param r: Proportion of frequency components to perturb (default is 0.5, meaning 50% of the frequency components will be perturbed)
    :param rng: Optional random number generator, if not provided, the global numpy random number generator will be used.

    :return: Perturbed time series, a 1D numpy array of the same length as the input series.
    """
    f = np.fft.rfft(series)
    f_perturbed = f.copy()
    frequencies = np.fft.fftfreq(len(series))

    # Calculate the number of frequency domain components that can be perturbed
    K = int(len(frequencies) * r)

    # Sample random perturbations for the real and imaginary parts of the frequency components
    alpha_real = sample_random_perturbation(
        K=K, min_alpha=min_alpha, max_alpha=max_alpha, rng=rng
    )
    alpha_imag = sample_random_perturbation(
        K=K, min_alpha=min_alpha, max_alpha=max_alpha, rng=rng
    )

    # Randomly select K frequency domain components for perturbation
    indices = np.random.choice(len(f_perturbed), size=K, replace=False)
    f_perturbed[indices] += alpha_real + 1j * alpha_imag

    # Perform inverse Fourier transform to restore the original time-domain signal
    perturbed_series = np.fft.irfft(f_perturbed).real

    return perturbed_series
