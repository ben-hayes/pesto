r"""The implementation of the CQT comes from the nnAudio repository: https://github.com/KinWaiCheuk/nnAudio
Due to conflicts between some versions of NumPy and nnAudio, we use the implementation as is instead of adding nnAudio
to the requirements of this project. Compared to the original implementation, some minor modifications have been done
in the code, however the behaviour remains the same.
"""
from typing import Optional

import numpy as np
import warnings
from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cqt_utils import *


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning(
            "You are using Kaiser window with beta factor "
            + str(window)
            + ". Correct behaviour not checked."
        )
    else:
        raise Exception(
            "The function get_window from scipy only supports strings, tuples and floats."
        )


def create_cqt_kernels(
        Q,
        fs,
        fmin: float,
        n_bins=84,
        bins_per_octave=12,
        norm=1,
        window="hann",
        fmax: Optional[float] = None,
        topbin_check=True,
        gamma=0,
        pad_fft=True
):
    """
    Automatically create CQT kernels in time domain
    """
    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(
                np.max(freqs)
            )
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_dispatch = get_window_dispatch(window, int(l), fftbins=True)
        sig = window_dispatch * np.exp(np.r_[-l // 2: l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start: start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start: start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs

class CQT(nn.Module):
    """
    This algorithm is using the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave.
    Then we keep downsampling the input audio by a factor of 2 to convoluting it with the
    small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled
    input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code
    from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        norm=True,
        basis_norm=1,
        window="hann",
        pad_mode="reflect",
        trainable_STFT=False,
        filter_scale=1,
        trainable_CQT=False,
        output_format="Magnitude",
        earlydownsample=True,
        verbose=True,
    ):

        super().__init__()

        self.norm = (
            norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        )
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.output_format = output_format
        self.earlydownsample = (
            earlydownsample  # TODO: activate early downsampling later if possible
        )

        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        # Creating lowpass filter and make it a torch tensor
        if verbose == True:
            print("Creating low pass filter ...", end="\r")
        start = time()
        lowpass_filter = torch.tensor(
            create_lowpass_filter(
                band_center=0.5, kernelLength=256, transitionBandwidth=0.001
            )
        )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer("lowpass_filter", lowpass_filter[None, None, :])

        if verbose == True:
            print(
                "Low pass filter created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        # print("n_octaves = ", self.n_octaves)

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (
            1 - 1 / bins_per_octave
        )  # Adjusting the top minium bins
        if fmax_t > sr / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, \
                              please reduce the n_bins".format(
                    fmax_t
                )
            )

        if (
            self.earlydownsample == True
        ):  # Do early downsampling if this argument is True
            if verbose == True:
                print("Creating early downsampling filter ...", end="\r")
            start = time()
            (
                sr,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(
                sr, hop_length, fmax_t, Q, self.n_octaves, verbose
            )

            self.register_buffer("early_downsample_filter", early_downsample_filter)
            if verbose == True:
                print(
                    "Early downsampling filter created, \
                            time used = {:.4f} seconds".format(
                        time() - start
                    )
                )
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        if verbose == True:
            print("Creating CQT kernels ...", end="\r")

        start = time()
        # print("Q = {}, fmin_t = {}, n_filters = {}".format(Q, self.fmin_t, n_filters))
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            sr,
            self.fmin_t,
            n_filters,
            bins_per_octave,
            norm=basis_norm,
            topbin_check=False,
        )

        # This is for the normalization in the end
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        self.frequencies = freqs

        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer("lenghts", lenghts)

        self.basis = basis
        fft_basis = fft(basis)[
            :, : self.n_fft // 2 + 1
        ]  # Convert CQT kenral from time domain to freq domain

        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(fft_basis.real)
        cqt_kernels_imag = torch.tensor(fft_basis.imag)

        if verbose == True:
            print(
                "CQT kernels created, time used = {:.4f} seconds".format(time() - start)
            )

        # print("Getting cqt kernel done, n_fft = ",self.n_fft)
        # Preparing kernels for Short-Time Fourier Transform (STFT)
        # We set the frequency range in the CQT filter instead of here.

        if verbose == True:
            print("Creating STFT kernels ...", end="\r")

        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(
            self.n_fft, window="ones", freq_scale="no"
        )
        wsin = kernel_sin * window
        wcos = kernel_cos * window

        wsin = torch.tensor(wsin)
        wcos = torch.tensor(wcos)

        if verbose == True:
            print(
                "STFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )

        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)
        else:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

        if trainable_CQT:
            cqt_kernels_real = nn.Parameter(
                cqt_kernels_real, requires_grad=trainable_CQT
            )
            cqt_kernels_imag = nn.Parameter(
                cqt_kernels_imag, requires_grad=trainable_CQT
            )
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

            # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format

        x = broadcast_dim(x)
        if self.earlydownsample == True:
            x = downsampling_by_n(
                x, self.early_downsample_filter, self.downsample_factor
            )
        hop = self.hop_length

        CQT = get_cqt_complex2(
            x,
            self.cqt_kernels_real,
            self.cqt_kernels_imag,
            hop,
            self.padding,
            wcos=self.wcos,
            wsin=self.wsin,
        )

        x_down = x  # Preparing a new variable for downsampling
        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)

            CQT1 = get_cqt_complex2(
                x_down,
                self.cqt_kernels_real,
                self.cqt_kernels_imag,
                hop,
                self.padding,
                wcos=self.wcos,
                wsin=self.wsin,
            )
            CQT = torch.cat((CQT1, CQT), 1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted top bins

        if normalization_type == "librosa":
            CQT *= torch.sqrt(self.lenghts.view(-1, 1, 1)) / self.n_fft
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2 / self.n_fft
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            # Getting CQT Amplitude
            return torch.sqrt(CQT.pow(2).sum(-1))

        elif output_format == "Complex":
            return CQT

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)

    def extra_repr(self) -> str:
        return "STFT kernel size = {}, CQT kernel size = {}".format(
            (*self.wcos.shape,), (*self.cqt_kernels_real.shape,)
        )



class CQTold(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it runs faster
    than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis, analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window should be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

    output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    Returns
    -------
    spectrogram : torch.Tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = CQT()
    >>> specs = spec_layer(x)
    """

    def __init__(
            self,
            sr=22050,
            hop_length=512,
            fmin=32.70,
            fmax=None,
            n_bins=84,
            bins_per_octave=12,
            filter_scale=1,
            norm=1,
            window="hann",
            center=True,
            pad_mode="reflect",
            trainable=False,
            output_format="Magnitude"
    ):

        super().__init__()

        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax
        )

        self.register_buffer("lenghts", lenghts)
        self.frequencies = freqs

        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)

        if trainable:  # NOTE: can't it be factorized?
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        normalization_type : str
            Type of the normalization. The possible options are: \n
            'librosa' : the output fits the librosa one \n
            'convolutional' : the output conserves the convolutional inequalities of the wavelet transform:\n
            for all p ϵ [1, inf] \n
                - || CQT ||_p <= || f ||_p || g ||_1 \n
                - || CQT ||_p <= || f ||_1 || g ||_p \n
                - || CQT ||_2 = || f ||_2 || g ||_2 \n
            'wrap' : wraps positive and negative frequencies into positive frequencies. This means that the CQT of a
            sinus (or a cosine) with a constant amplitude equal to 1 will have the value 1 in the bin corresponding to
            its frequency.
        """
        output_format = output_format or self.output_format

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)

            x = padding(x)

        # CQT
        CQT_real = F.conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = -F.conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        if normalization_type == "librosa":
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            margin = 1e-8 if self.trainable else 0
            return torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2) + margin)

        elif output_format == "Complex":
            return torch.stack((CQT_real, CQT_imag), -1)

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)


class HarmonicCQT(nn.Module):
    r"""Harmonic CQT layer, as described in Bittner et al. (20??)"""
    def __init__(
            self,
            harmonics,
            sr: int = 22050,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: Optional[float] = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True
    ):
        super(HarmonicCQT, self).__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList([
            CQT(sr=sr, hop_length=hop_length, fmin=h*fmin, fmax=fmax, n_bins=n_bins,
                bins_per_octave=12*bins_per_semitone, output_format="Complex")
            for h in harmonics
        ])

    def forward(self, audio_waveforms: torch.Tensor):
        r"""Converts a batch of waveforms into a batch of HCQTs.

        Args:
            audio_waveforms (torch.Tensor): Batch of waveforms, shape (batch_size, num_samples)

        Returns:
            Harmonic CQT, shape (batch_size, num_harmonics, num_freqs, num_timesteps, 2)
        """
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)
