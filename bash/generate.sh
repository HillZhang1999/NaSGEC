CUDA_DEVICE=0
BEAM=12
N_BEST=1
SEED=2022
FAIRSEQ_DIR=../SynGEC/src/src_syngec/fairseq-0.10.2/fairseq_cli

TEST_DIR=/mnt/nas_alinlp/zuyi.bzy/zhangyue/NaSGEC/data/test # 测试集路径
INPUT_FILE=$TEST_DIR/input.txt
MODEL_PATH=../models/real_learner_bart_CGEC.pt # 模型路径
PROCESSED_DIR=../data/dict
OUTPUT_DIR=../results/test

# tokenizing
if [ ! -f $INPUT_FILE".char" ]; then
  echo "Tokenizing..."
  cd ../utils
  python segment_bert.py <$INPUT_FILE >$INPUT_FILE".char"
  cd -
fi

mkdir -p $OUTPUT_DIR
cp $INPUT_FILE $OUTPUT_DIR/input.txt
INPUT_FILE=$INPUT_FILE".char"
cp $INPUT_FILE $OUTPUT_DIR/input.char

echo "Generating..."
SECONDS=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR \
  --user-dir ../SynGEC/src/src_syngec/syngec_model \
  --task syntax-enhanced-translation \
  --path ${MODEL_PATH} \
  --beam ${BEAM} \
  --nbest ${N_BEST} \
  -s src \
  -t tgt \
  --buffer-size 10000 \
  --batch-size 32 \
  --num-workers 12 \
  --log-format tqdm \
  --remove-bpe \
  --fp16 \
  --output_file $OUTPUT_DIR/output.nbest \
  <$OUTPUT_DIR/input.char

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/output.nbest | grep "^D-" | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 >$OUTPUT_DIR/output.char
sed -i '$d' $OUTPUT_DIR/output.char
cat $OUTPUT_DIR/output.char | python -c "import sys; x = sys.stdin.readlines(); x = '\n'.join([''.join([tok[2:] if len(tok) > 2 and tok[:2] == '##' else tok for tok in sent.split()]) for sent in x]); print(x)" >$OUTPUT_DIR/output.txt # 最终预测结果
