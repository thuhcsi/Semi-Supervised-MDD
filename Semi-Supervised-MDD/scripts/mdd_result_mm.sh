cwd=$(pwd)

. ./../../path.sh




cd /mnt/d/projects/Semi-Supervised-MDD/results/mm


#step1 计算PER
compute-wer  --text --mode=present ark:tran_seq ark:decode_seq > per || exit 1;

#step2 计算Recall and Precision
# note : no sil
align-text ark:tran_seq  ark:decode_seq ark,t:- | $cwd/wer_per_utt_details.pl > ref_human_detail
