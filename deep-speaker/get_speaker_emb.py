import os, sys, struct, glob, random, argparse
import numpy as np
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES, NUM_FBANKS, EMBEDDING_DIM
from conv_models import DeepSpeakerModel
from tqdm import tqdm


def get_filename(outputdir, s):
	if s < 10:
		return os.path.join(outputdir, 'speaker_00' + str(s) + '_mean.npy')
	elif s < 100:
		return os.path.join(outputdir, 'speaker_0' + str(s) + '_mean.npy')
	else:
		return os.path.join(outputdir, 'speaker_' + str(s) + '_mean.npy')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputdir', required=True, help='''--inputdir=<mels_dir> location of the input mels''')
	parser.add_argument('--outdir', required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--speakerid', required=True, help='Speaker ID. If two numbers are provided separated from comma, e.g. <spk1,spk2>, extracts from spk1 to spk2 (default=\'0\')')
	parser.add_argument('--embed_out', default=False, help='''--embed_out=<boolean> whether to output embeddings or not (default=False).''')
	parser.add_argument('--checkpoint_file', required=True, help='''--checkpoint_file=<file.h5> location of the checkpoint.''')

	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	speakerid	= args.speakerid.split(",")
	embed_out	= args.embed_out
	checkpoint	= args.checkpoint_file

	np.random.seed(123)
	random.seed(123)

	model = DeepSpeakerModel()
	model.m.load_weights(checkpoint)

	os.makedirs(outputdir, exist_ok=True)
	if embed_out:
		embed_out_dir = os.path.join(outputdir, 'embeddings')
		os.makedirs(embed_out_dir, exist_ok=True)

	for s in range(int(speakerid[0]),int(speakerid[-1])+1):
		filelist = glob.glob(inputdir + '/' + str(s) + '_*.npy')
		if len(filelist) >= 1:
			D = []
			for i, file in enumerate(tqdm(filelist)):
				base = os.path.basename(file)
				mfcc = sample_from_mfcc(np.load(file), NUM_FRAMES)
				mfcc = mfcc.reshape([1, NUM_FRAMES, NUM_FBANKS, 1])
				dj = model.m.predict(mfcc)
				D += dj.tolist()
				if embed_out:
					np.save(os.path.join(embed_out_dir, os.path.basename(file)), dj)
			D = np.asarray(D)
			dvector = np.mean(D, axis=0)
		else:
			D = np.zeros([1, EMBEDDING_DIM])
			dvector = np.zeros([1, EMBEDDING_DIM])
		out_dvector_file = get_filename(outputdir, s)
		print(f'\nd-vector for speaker {s}:')
		print(dvector)
		print(f'Dimension of the D matrix : {D.shape}')
		print(f'Dimension of the d vector : {dvector.shape}')
		print(f'Output file: {out_dvector_file}\n')
		np.save(out_dvector_file, dvector)


if __name__ == "__main__":
	main()
