import os, argparse, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import spatial
from speakers_ptBR import params


def main():
	# input argument processing
	parser = argparse.ArgumentParser()
	parser.add_argument('--emb_file', required=True, help='''--emb_file=<speaker_embedding_file> ''')
	parser.add_argument('--dim', default=2, help='''--dim=<dimension_of_plot(2|3)> 2D or 3D plot''')
	parser.add_argument('--mode', default='sne', help='''--mode=<mode> dimension reduction approach (default: 'sne')''')
	parser.add_argument('--mls', default=False, help='''--mls=<boolean> whether to plot MLS speakers as well (default: False)''')	
	parser.add_argument('--libritts', default=False, help='''--libritts=<boolean> whether to plot LIBRITTS speakers as well (default: False)''')
	parser.add_argument('--savefig', default=None, help='''--savefig=<file.png> file to save figure (default: None)''')
	args = parser.parse_args()
	emb_file	= args.emb_file
	mode		= args.mode
	savefig		= args.savefig
	dim		= int(args.dim)

	speakers	=	params.speakers
	if args.mls:
		speakers += params.mls_speakers
	if args.libritts:
		speakers += params.libritts_speakers

	print(f'\nEmbedding file: {emb_file}\n')
	print(f'Plotting mode: {mode}\n')
	print(f'Embedding dimension: {emb_file}\n')
	print(f'Dimension of the plot: {dim}\n')

	emb = np.load(emb_file)
	print(f'These indices have zero embeddings: {np.where(np.sum(emb,axis=1)==0)}\n')
	if mode == 'sne':
		X = TSNE(n_components=dim,perplexity=5,learning_rate=100.,init='pca').fit_transform(emb)
		delta = 0.5
	elif mode == 'pca':
		X = PCA(n_components=2).fit_transform(emb)
		delta = 0.005
	else:
		raise ValueError(f'Mode {mode} not supported.')

	fig = plt.figure(figsize=(14,10))
	if dim == 2:
		ax = fig.add_subplot(111)
	elif dim == 3:
		ax = fig.add_subplot(111,projection='3d')
	else:
		raise ValueError(f'Dimension {dim} not supported.')

	print(f'Shape of the 2D embedding: {X.shape}')

	for s, i, c in speakers:
		if c == 1:
			colors = 'r'
		elif c == 0:
			colors = 'b'
		else:
			colors = 'g'
		if dim == 2:
			ax.scatter(X[i:i+1, 0], X[i:i+1, 1], color=colors)
			ax.text(X[i:i+1, 0]+delta, X[i:i+1, 1]+delta, s, fontsize=9)
		else:
			ax.scatter(X[i:i+1, 0], X[i:i+1, 1], X[i:i+1, 2], color=colors)
			ax.text(X[i:i+1, 0]+delta, X[i:i+1, 1]+delta, X[i:i+1, 2]+delta, s, fontsize=9)

	plt.legend(fontsize=10)
	plt.title('Speaker embeddings')
	if savefig:
		plt.savefig(savefig.replace('.png','') + '.png')
	plt.show()


if __name__ == "__main__":
	main()






