CUDA_VISIBLE_DEVICES=1 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take1 \
	--speakerid 0,1 \
	--model_path model_output/ptBR/t01/ptBR_train_v01-July-02-2021_11+33PM-debug/best_model.pth.tar

CUDA_VISIBLE_DEVICES=0 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take2 \
	--speakerid 0,1 \
	--model_path model_output/ptBR/t01/ptBR_train_v01-July-02-2021_11+33PM-debug/best_model.pth.tar

CUDA_VISIBLE_DEVICES=1 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder_v02.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take3 \
	--speakerid 0,1 \
	--model_path model_output/ptBR/t02/ptBR_train_v01-July-10-2021_10+33AM-debug/best_model.pth.tar

CUDA_VISIBLE_DEVICES=0 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder_v02.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take4 \
	--speakerid 0,1 \
	--model_path model_output/ptBR/t02/ptBR_train_v01-July-10-2021_10+33AM-debug/best_model.pth.tar

CUDA_VISIBLE_DEVICES=1 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take5 \
	--speakerid 0,1 \
	--model_path TTS/bin/MozillaTTSOutput/checkpoints/ptBR_train_v01-July-02-2021_06+56PM-debug/best_model.pth.tar

CUDA_VISIBLE_DEVICES=1 python TTS/bin/get_speaker_embeddings.py \
	--config_path TTS/speaker_encoder/ptBR_speaker_encoder.json \
	--inputdir ../deep-speaker/portuguese/pre-training/audio-fbanks/ \
	--outdir take6 \
	--speakerid 0,1 \
	--model_path TTS/bin/MozillaTTSOutput/checkpoints/ptBR_train_v01-July-02-2021_06+56PM-debug/best_model.pth.tar

for f in take{1,2,3,4,5,6}/speaker_000_mean.npy; do diff $f dvectors/speaker_000_mean.npy; done


