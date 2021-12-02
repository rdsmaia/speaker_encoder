import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	#Hardware setup: Default supposes user has only one GPU: "/gpu:0" (Both Tacotron and WaveNet can be trained on multi-GPU: data parallelization)
	#Synthesis also uses the following hardware parameters for multi-GPU parallel synthesis.
	tacotron_num_gpus = 1, #Determines the number of gpus in use for Tacotron training.
	split_on_cpu = True, #Determines whether to split data on CPU or on first GPU. This is automatically True when more than 1 GPU is used. 
		#(Recommend: False on slow CPUs/Disks, True otherwise for small speed boost)
	###########################################################################################################################################

	#Audio parameters
	num_mels = 128, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value

	#train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
	clip_mels_length = True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
	max_mel_frames = 1300,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False, #Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 220, #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
	win_size = 1100, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
	frame_shift_ms = 10., #Can replace hop_size parameter. (Recommended: 12.5)
	magnitude_power = 2., #The power of the spectrogram magnitude (1. for energy, 2. for power)
	predict_linear = False, #Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)
	save_audio = False,

	#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	trim_fft_size = 2048, #Trimming window size
	trim_hop_size = 512, #Trimmin hop length
	trim_top_db = 40, #Trimming db difference from reference db (smaller==harder trim.)
	wavenet_pad_sides = 1, #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
																										  #not too small for fast convergence)
	#Contribution by @begeekmyfriend
	#Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
	preemphasize = True, #whether to apply filter
	preemphasis = 0.97, #filter coefficient.
        apply_postfiltering = False,
        pf_coef = 1.4,
	warp_scale = 'Mel',

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 0, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax = 11025, #To be increased/reduced depending on data.

	#Griffin Lim
	power = 2.0, #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters = 60, #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
	GL_on_GPU = False, #Whether to use G&L GPU version as part of tensorflow graph. (Usually much faster than CPU but slightly worse quality too).
	###########################################################################################################################################
	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
