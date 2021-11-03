import os, sys, struct, glob, random, argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES, NUM_FBANKS
from conv_models import DeepSpeakerModel
from tqdm import tqdm


def plot_imshow(scores, path, title=None):

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        im = ax.imshow(
                scores,
                aspect='auto',
                origin='lower',
                interpolation='none')
        fig.colorbar(im, ax=ax)
        plt.xlabel('speaker')
        plt.title(title)
        plt.ylabel('speaker')
        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.close()


def plot_heatmap(scores, path, lspeakers, title=None):
	nspeakers = len(scores)
	df_cm = pd.DataFrame(scores, range(nspeakers), range(nspeakers))
	sn.set(font_scale=0.1) # for label size
	sn.heatmap(
		df_cm,
		vmin=-1,
		vmax=1,
		annot=True,
		annot_kws={"size": 1},
		linewidths=.1,
		xticklabels=lspeakers,
		yticklabels=lspeakers) # font size
	plt.xlabel('speaker')
	plt.title(title)
	plt.ylabel('speaker')
	plt.tight_layout()
	plt.savefig(path, format='png')
	plt.close()


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
	parser.add_argument('--dvectors', required=True, help='''--dvectors=<dvectors_dir> location of the centroids''')
	parser.add_argument('--outdir', required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--speakerid', required=True, help='Speaker ID. If two numbers are provided separated from comma, e.g. <spk1,spk2>, extracts from spk1 to spk2 (default=\'0\')')
	parser.add_argument('--embed_out', type=bool, default=False, help='''--embed_out=<boolean> whether to output embeddings or not (default=False).''')
	parser.add_argument('--checkpoint_file', required=True, help='''--checkpoint_file=<file.h5> location of the checkpoint.''')
	parser.add_argument('--modelname', type=str, required=True, help='''--modelname=<file.png> for saving the figfile name.''')
	parser.add_argument('--heatmap', type=bool, default=False, help='''--heatmap=<boolean> whether to use heatmap or not for the plot.''')

	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	dvector_dir	= args.dvectors
	speakerid	= args.speakerid.split(",")
	embed_out	= args.embed_out
	checkpoint	= args.checkpoint_file
	embed_dim	= 64
	modelname	= args.modelname
	heatmap		= args.heatmap

	np.random.seed(123)
	random.seed(123)

	model = DeepSpeakerModel()
	model.m.load_weights(checkpoint)

	os.makedirs(outputdir, exist_ok=True)
	if embed_out:
		embed_out_dir = os.path.join(outputdir, 'embeddings')
		os.makedirs(embed_out_dir, exist_ok=True)

	nspeakers = int(speakerid[-1]) + 1
	C = np.zeros((nspeakers, embed_dim))
	lspeakers = []
	for k in range(int(speakerid[0]),int(speakerid[-1])+1):
		dvector_file = get_filename(dvector_dir, k)
		ck = np.load(dvector_file)
		C[k,:] = ck.reshape(1,-1)
		lspeakers.append('spk' + str(k))
		print(f'\nd-vector file for speaker {k}: {dvector_file}')
		print(f'd-vector: {ck}')
		print(f'Dimension of the d vector : {ck.shape}')
	print(f'Shape of the centroid matrix: {C.shape}\n')

	print('Reading embeddings and calculating distances...')
	scores = []
	for k in range(int(speakerid[0]),int(speakerid[-1])+1):
		filelist = glob.glob(inputdir + '/' + str(k) + '_*.npy')
		if len(filelist) >= 1:
			S = []
			for i, file in enumerate(tqdm(filelist)):
				base = os.path.basename(file)
				mfcc = sample_from_mfcc(np.load(file), NUM_FRAMES)
				mfcc = mfcc.reshape([1, NUM_FRAMES, NUM_FBANKS, 1])
				si = model.m.predict(mfcc)
				S += si.tolist()
			S = np.asarray(S)
		else:
			S = np.zeros((1,embed_dim))
		print(f'Shape of the embedding matrix for speaker {k}: {S.shape}')
		scores.append(np.squeeze(np.mean(np.matmul(S, C.T), axis=0)).tolist())
	print(lspeakers)
	print(scores)

#	if heatmap:
#		plot_heatmap(scores, modelname+'.png', lspeakers, title='ptBR')
#	else:
	plot_imshow(np.asarray(scores), modelname+'.png', title='ptBR')



if __name__ == "__main__":
	main()
