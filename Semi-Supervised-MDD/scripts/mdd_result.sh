cwd=$(pwd)

. ./path.sh

config_name=$1
data_dir=$2
segment=$3
output_dir=$4
result_dir=$output_dir/$config_name


cd $result_dir
cp $data_dir/$segment/phn_text ./
cp $data_dir/$segment/transcript_phn_text ./

#step1 计算PER
compute-wer  --text --mode=present ark:phn_text ark:decode_seq > per || exit 1;

#step2 计算Recall and Precision
# note : no sil
align-text ark:transcript_phn_text  ark:phn_text ark,t:- | $cwd/Semi-Supervised-MDD/scripts/wer_per_utt_details.pl > ref_human_detail
align-text ark:phn_text  ark:decode_seq ark,t:- | $cwd/Semi-Supervised-MDD/scripts/wer_per_utt_details.pl > human_our_detail
align-text ark:transcript_phn_text  ark:decode_seq ark,t:- | $cwd/Semi-Supervised-MDD/scripts/wer_per_utt_details.pl > ref_our_detail
python3 $cwd/Semi-Supervised-MDD/scripts/ins_del_sub_cor_analysis_without_sil.py
cat mdd_result