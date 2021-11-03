import os, sys, struct, glob, random, argparse
import numpy as np

import torch
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
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
	parser.add_argument('--dvectors', required=True, help='''--dvectors=<dvectors_dir> location of the centroids''')
	parser.add_argument('--outdir', required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--speakerid', required=True, help='Speaker ID. If two numbers are provided separated from comma, e.g. <spk1,spk2>, extracts from spk1 to spk2 (default=\'0\')')
	parser.add_argument('--embed_out', type=bool, default=False, help='''--embed_out=<boolean> whether to output embeddings or not (default=False).''')
	parser.add_argument('--checkpoint_file', required=True, help='''--checkpoint_file=<file.h5> location of the checkpoint.''')
	parser.add_argument('--scoresfile', type=str, required=True, help='''--scoresfile=<scores.npy> score file.''')
	parser.add_argument('--config_path', type=str, required=True, help='Path to config file for training.', )
	parser.add_argument('--use_cuda', type=bool, help='flag to set cuda.', default=True)

	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	dvector_dir	= args.dvectors
	speakerid	= args.speakerid.split(",")
	embed_out	= args.embed_out
	checkpoint	= args.checkpoint_file
	embed_dim	= 64
	scoresfilename	= args.scoresfile
	config		= args.config_path

        # define Encoder model
	c = load_config(args.config_path)
	model = SpeakerEncoder(**c.model)
	model.load_state_dict(torch.load(checkpoint)['model'])
	model.eval()
	if args.use_cuda:
		model.cuda()

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
	filelist = glob.glob(inputdir + '/mel-LJ*.npy')
	assert len(filelist) >= 1
	S = []
	for file in tqdm(filelist):
		base = os.path.basename(file)
		mel_spec = np.load(file)
		mel_spec = torch.FloatTensor(mel_spec[None, :, :])
		if args.use_cuda:
			mel_spec = mel_spec.cuda()
		si = model.compute_embedding(mel_spec).detach().cpu().numpy()
		if embed_out:
			np.save(os.path.join(embed_out_dir, base), si)
		S += si.tolist()
	S = np.asarray(S)
	print(f'Shape of the embedding matrix for speaker {k}: {S.shape}')

        # calculate scores: scores (num_sent x num_centroids) = S (num_sent x embed_dim) * C^T (embed_dim x num_centroids)
        # and take the mean across sentences: scores_mean (1 x num_centroids)
	scores = np.mean(np.matmul(S, C.T), axis=0)

	print(lspeakers)
	print(scores)

	np.save(scoresfilename, scores)


if __name__ == "__main__":
	main()
