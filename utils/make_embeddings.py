import numpy as np
import argparse, os
from numpy import linalg as LA

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputdir', required=True, help='''--inputdir=<dvectors_dir> location of the input dvectors''')
	parser.add_argument('--outfile', required=True, help='''--outfile=<out_embedding_file> embedding file''')
	parser.add_argument('--norm', default=False, help='''--norm=<boolean> whether to normalize the dvectors or not (default: False).''')
	parser.add_argument('--num_speakers', default='100', help='''--num_speakers=<num_speakers> number of speakers to be considered (default: 100).''')
	parser.add_argument('--embed_dim', default='64', help='''--embed_dim=<embed_dim> dimension of the embedding (default: 64).''')
	args = parser.parse_args()
	inputdir	= args.inputdir
	outfile		= args.outfile
	norm		= args.norm
	num_speakers	= int(args.num_speakers)
	embed_dim	= int(args.embed_dim)
	print(f'\nInput dvector directory: {inputdir}')
	print(f'Ourput file: {outfile}')
	print(f'Normalize: {norm}')
	print(f'Number of speakers: {num_speakers}')
	print(f'Embedding dimension: {embed_dim}\n')

	D = np.zeros([num_speakers, embed_dim])

	for i in range(num_speakers):
		if i < 10:
			j = '00' + str(i)
		elif i < 100:
			j = '0' + str(i)
		else:
			j = str(i)
		dvector_file = os.path.join(inputdir, 'speaker_' + j + '_mean.npy')
		if os.path.isfile(dvector_file):
			d=np.load(dvector_file)
			if norm is True:
				D[i, :] = d / LA.norm(d)
			else:
				D[i, :] = d
	print(f'D-vector norm: {LA.norm(D, axis=1)}')
	np.save(outfile, D)


if __name__ == "__main__":
	main()




