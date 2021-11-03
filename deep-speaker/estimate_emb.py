import os, sys, struct, glob, random, argparse
import numpy as np
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES, NUM_FBANKS, EMBEDDING_DIM
from conv_models import DeepSpeakerModel
from tqdm import tqdm


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputdir', required=True, help='--inputdir=<mels_dir> location of the input mels')
	parser.add_argument('--outdir', required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--checkpoint_file', required=True, help='--checkpoint_file=<file.h5> location of the checkpoint')
	parser.add_argument('--output_mean', type=str, default=None, help='--output_mean=<mean_file.npy> whether or not to output mean.')
	parser.add_argument('--deepzen_model', type=bool, default=False, help='--deepzen_mozel=<bool> whether symmetric mels should be rescaled to [0, 4) (default: False)')
	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	checkpoint	= args.checkpoint_file
	outmean		= args.output_mean
	deepzen_model	= args.deepzen_model
	print(f'Rescale from [-4,4) to [0,4): {deepzen_model}')

	np.random.seed(123)
	random.seed(123)

	model = DeepSpeakerModel()

	model.m.load_weights(checkpoint)
	os.makedirs(outputdir, exist_ok=True)

	filelist = glob.glob(inputdir + '/*.npy')
	if outmean is not None:
		D = np.empty([1, EMBEDDING_DIM])
	for i, file in enumerate(tqdm(filelist)):
		out_dvector_file = os.path.join(outputdir, os.path.basename(file))
		mfcc = sample_from_mfcc(np.load(file), NUM_FRAMES)
		if deepzen_model:
			# NOTE: changes rescaling from [-4,4) -> [0,4)
			mfcc = (mfcc + 4.) / 2.
		mfcc = mfcc.reshape([1, NUM_FRAMES, NUM_FBANKS, 1])
		dj = model.m.predict(mfcc)
		np.save(out_dvector_file, dj)
		if outmean is not None:
			D = np.concatenate((D,dj), axis=0)
	if outmean is not None:
		np.save(outmean, np.mean(D, axis=0))

if __name__ == "__main__":
	main()
