import argparse, glob, os

import numpy as np
from tqdm import tqdm

import torch
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.io import load_config


def get_filename(outputdir, s):
	if s < 10:
		return os.path.join(outputdir, 'speaker_00' + str(s) + '_mean.npy')
	elif s < 100:
		return os.path.join(outputdir, 'speaker_0' + str(s) + '_mean.npy')
	else:
		return os.path.join(outputdir, 'speaker_' + str(s) + '_mean.npy')


def main():
	parser = argparse.ArgumentParser(
	description='Compute embedding vectors for each wav file in a dataset. If "target_dataset" is defined, it generates "speakers.json" necessary for training a multi-speaker model.')
	parser.add_argument('--model_path', type=str, required=True, help='Path to model outputs (checkpoint, tensorboard etc.).')
	parser.add_argument('--config_path', type=str, required=True, help='Path to config file for training.', )
	parser.add_argument('--use_cuda', type=bool, help='flag to set cuda.', default=True)
	parser.add_argument('--inputdir', type=str, required=True, help='''--inputdir=<mels_dir> location of the input mels''')
	parser.add_argument('--outdir', type=str, required=True, help='--outdir=<out_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--speakerid', required=True, help='Speaker ID. If two numbers are provided separated from comma, e.g. <spk1,spk2>, extracts from spk1 to spk2 (default=\'0\')')

	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	speakerid	= args.speakerid.split(",")
	checkpoint	= args.model_path
	embed_dim	= 64

	# define Encoder model
	c = load_config(args.config_path)
	model = SpeakerEncoder(**c.model)
	model.load_state_dict(torch.load(checkpoint)['model'])
	model.eval()
	if args.use_cuda:
		 model.cuda()

	os.makedirs(outputdir, exist_ok=True)

	for s in range(int(speakerid[0]), int(speakerid[-1])+1):

		filelist = glob.glob(inputdir + '/' + str(s) + '_*.npy')

		if len(filelist) >= 1:
			D = []
			for mel_file in tqdm(filelist):
				mel_spec = np.load(mel_file).astype(np.float32)
				mel_spec = torch.FloatTensor(mel_spec[None, :, :])
				if args.use_cuda:
					mel_spec = mel_spec.cuda()
				embedd = model.compute_embedding(mel_spec)
				embedd = embedd.detach().cpu().numpy()
				D += embedd.tolist()
			D = np.asarray(D)
			dvector = np.mean(D, axis=0)
		else:
			D = np.zeros([1, embed_dim])
			dvector = np.zeros([1, embed_dim])

		out_dvector_file = get_filename(outputdir, s)

		print(f'\nd-vector for speaker {s}:')
		print(dvector)
		print(f'Dimension of the D matrix : {D.shape}')
		print(f'Output file: {out_dvector_file}\n')
		np.save(out_dvector_file, dvector)


if __name__ == "__main__":
	main()
