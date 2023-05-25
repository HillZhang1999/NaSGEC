# Preprocess data for fine-tuning

FAIRSEQ_DIR=../SynGEC/src/src_syngec/fairseq-0.10.2/fairseq_cli
DATA_DIR=../data/NaSGEC-Thesis
PROCESSED_DIR=../preprocess/NaSGEC-Thesis
mkdir -p $PROCESSED_DIR
WORKER_NUM=64
SYNTAX_DICT=../data/dict/dict.label0.txt

TRAIN_SRC_FILE=$DATA_DIR/train/src.txt
TRAIN_TGT_FILE=$DATA_DIR/train/tgt.txt
VALID_SRC_FILE=$DATA_DIR/dev/src.txt
VALID_TGT_FILE=$DATA_DIR/dev/tgt.txt

# tokenizing
if [ ! -f $TRAIN_SRC_FILE".char" ]; then
  echo "Tokenizing..."
  cd ../utils
  python segment_bert.py <$TRAIN_SRC_FILE >$TRAIN_SRC_FILE".char"
  python segment_bert.py <$TRAIN_TGT_FILE >$TRAIN_TGT_FILE".char"
  python segment_bert.py <$VALID_SRC_FILE >$VALID_SRC_FILE".char"
  python segment_bert.py <$VALID_TGT_FILE >$VALID_TGT_FILE".char"
  cd -
fi

cp $TRAIN_SRC_FILE".char" $PROCESSED_DIR/train.char.src
cp $TRAIN_TGT_FILE".char" $PROCESSED_DIR/train.char.tgt
cp $VALID_SRC_FILE".char" $PROCESSED_DIR/valid.char.src
cp $VALID_TGT_FILE".char" $PROCESSED_DIR/valid.char.tgt
mkdir -p $PROCESSED_DIR/bin

echo "Preprocessing..."

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
  --user-dir ../ \
  --task syntax-enhanced-translation \
  --trainpref $PROCESSED_DIR/train.char \
  --validpref $PROCESSED_DIR/valid.char \
  --destdir $PROCESSED_DIR/bin \
  --workers $WORKER_NUM \
  --labeldict $SYNTAX_DICT \
  --srcdict ../data/dict/dict.src.txt \
  --tgtdict ../dict/dict/dict.src.txt

echo "Finished!"

######## Finetuning ########
SEED=42
PRETRAIN_MODEL_PATH=../model/pseudo_native_bart_zh.pt
MODEL_DIR=../model/pseudo_native_bart_zh_finetuned_with_NaSGEC_Thesis/$SEED

mkdir -p $MODEL_DIR/src
cp ./finetune.sh $MODEL_DIR

CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=INFO nohup python -u $FAIRSEQ_DIR/train.py $PROCESSED_DIR/bin \
  --save-dir $MODEL_DIR \
  --user-dir ../ \
  --task syntax-enhanced-translation \
  --arch syntax_enhanced_bart_large \
  --finetune-from-model $PRETRAIN_MODEL_PATH \
  --skip-invalid-size-inputs-valid-test \
  --max-tokens 1024 \
  --update-freq 1 \
  --optimizer adam \
  --lr 1e-05 \
  --max-source-positions 512 \
  --max-target-positions 512 \
  --warmup-updates 0 \
  -s src \
  -t tgt \
  --lr-scheduler polynomial_decay \
  --clip-norm 1.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --dropout 0.3 \
  --share-all-embeddings \
  --adam-betas '(0.9,0.999)' \
  --log-format tqdm \
  --find-unused-parameters \
  --fp16 \
  --max-epoch 100 \
  --patience 10 \
  --seed $SEED >MODEL_DIR/nohup.log 2>&1 &

wait
