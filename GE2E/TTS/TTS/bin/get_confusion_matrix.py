import os, sys, struct, glob, random, argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
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
#	parser.add_argument('--outdir', required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--speakerid', required=True, help='Speaker ID. If two numbers are provided separated from comma, e.g. <spk1,spk2>, extracts from spk1 to spk2 (default=\'0\')')
#	parser.add_argument('--embed_out', type=bool, default=False, help='''--embed_out=<boolean> whether to output embeddings or not (default=False).''')
	parser.add_argument('--checkpoint_file', required=True, help='''--checkpoint_file=<file.h5> location of the checkpoint.''')
	parser.add_argument('--modelname', type=str, required=True, help='''--modelname=<file.png> for saving the figfile name.''')
	parser.add_argument('--heatmap', type=bool, default=False, help='''--heatmap=<boolean> whether to use heatmap or not for the plot.''')
	parser.add_argument('--config_path', type=str, required=True, help='Path to config file for training.', )
	parser.add_argument('--use_cuda', type=bool, help='flag to set cuda.', default=True)

	args = parser.parse_args()
	inputdir	= args.inputdir
#	outputdir	= args.outdir
	dvector_dir	= args.dvectors
	speakerid	= args.speakerid.split(",")
#	embed_out	= args.embed_out
	checkpoint	= args.checkpoint_file
	embed_dim	= 64
	modelname	= args.modelname
	heatmap		= args.heatmap
	config		= args.config_path

        # define Encoder model
	c = load_config(args.config_path)
	model = SpeakerEncoder(**c.model)
	model.load_state_dict(torch.load(checkpoint)['model'])
	model.eval()
	if args.use_cuda:
		model.cuda()

#	os.makedirs(outputdir, exist_ok=True)
#	if embed_out:
#		embed_out_dir = os.path.join(outputdir, 'embeddings')
#		os.makedirs(embed_out_dir, exist_ok=True)

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
				mel_spec = np.load(file)
				mel_spec = torch.FloatTensor(mel_spec[None, :, :])
				if args.use_cuda:
					mel_spec = mel_spec.cuda()
				si = model.compute_embedding(mel_spec).detach().cpu().numpy()
				S += si.tolist()
			S = np.asarray(S)
		else:
			S = np.zeros((1,embed_dim))
		print(f'Shape of the embedding matrix for speaker {k}: {S.shape}')
		scores.append(np.squeeze(np.mean(np.matmul(S, C.T), axis=0)).tolist())
	print(lspeakers)
	print(scores)
	np.save(modelname + '_scores.npy', np.asarray(scores))

#	if heatmap:
#		plot_heatmap(scores, modelname+'.png', lspeakers, title='ptBR')
#	else:
	plot_imshow(np.asarray(scores), modelname+'.png', title=modelname)



if __name__ == "__main__":
	main()
