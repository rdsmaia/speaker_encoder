import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

import torch
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.io import load_config


def main():
	parser = argparse.ArgumentParser(
	description='Compute embedding vectors for each wav file in a dataset. If "target_dataset" is defined, it generates "speakers.json" necessary for training a multi-speaker model.')
	parser.add_argument('--model_path', type=str, required=True, help='--model_path=<file.pth.tar> Path to model outputs (checkpoint, tensorboard etc.).')
	parser.add_argument('--config_path', type=str, required=True, help='--config_path=<config.json> Path to config file for training.')
	parser.add_argument('--use_cuda', type=bool, default=True, help='--use_cude=<boolean> flag to set cuda (default: True).')
	parser.add_argument('--inputdir', type=str, required=True, help='--inputdir=<mels_dir> location of the input mels')
	parser.add_argument('--outdir', type=str, required=True, help='--outdir=<output_dir> place to store generate dvector and embeddings.')
	parser.add_argument('--output_mean', type=str, default=None, help='--output_mean=<mean_file.npy> whether or not to output mean.')
	parser.add_argument('--deepzen_model', type=bool, default=False, help='--deepzen_model=<boolean> whether symmetric spectrograms should be rescaled to [0,4) (default: False).')

	args = parser.parse_args()
	inputdir	= args.inputdir
	outputdir	= args.outdir
	checkpoint	= args.model_path
	outmean		= args.output_mean
	embed_dim	= 64
	deepzen_model	= args.deepzen_model

	# define Encoder model
	c = load_config(args.config_path)
	model = SpeakerEncoder(**c.model)
	model.load_state_dict(torch.load(checkpoint)['model'])
	model.eval()
	if args.use_cuda:
		model.cuda()

	os.makedirs(outputdir, exist_ok=True)

	filelist = glob.glob(inputdir + '/*.npy')
	assert len(filelist) >= 1

	if outmean is not None:
		D = []
	for mel_file in tqdm(filelist):
		mel_spec = np.load(mel_file).astype(np.float32)
		if deepzen_model:
			# recale spectrograms from [-4,4) -> [0,4)
			mel_spec = (mel_spec + 4.) / 2.
		mel_spec = torch.FloatTensor(mel_spec[None, :, :])
		if args.use_cuda:
			mel_spec = mel_spec.cuda()
		embedd = model.compute_embedding(mel_spec)
		embedd = embedd.detach().cpu().numpy()
		np.save(os.path.join(outputdir, os.path.basename(mel_file)), embedd)
		if outmean is not None:
			D += embedd.tolist()
	if outmean is not None:
		np.save(outmean, np.mean(np.asarray(D), axis=0))


if __name__ == "__main__":
	main()
