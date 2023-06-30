# 基于样本选择和类别平衡损失的动量伪标签错误发音检测与诊断

伪标签法半监督学习可以让 MDD 模型在有少量标注数据的情况下较高效地利用无标注数据来提高模型能力，但这种方法的效果取决于伪标签的质量，本方法一方面通过针对无标注数据的样本选择策略过滤因领域迁移而引入的低质量伪标签，另一方面通过焦点连接时序分类（Focal CTC）损失函数使模型的伪标签不容易受类别不平衡问题影响而过拟合到高频音素上。通过对伪标签质量的改善，本方法可以提高对无标注数据的利用效率，从而改进模型的 MDD 表现。

## 环境设置

```shell
bash ./setup.sh
```

另外需要克隆并编译kaldi，把kaldi项目路径填入path.sh中的local_kaldi_path。

## 数据准备

```shell
python3 ./Semi-Supervised-MDD/scripts/timit_downsampling.py --raw_l2_arctic_dir $raw_l2_arctic_dir --output_dir $l2_arctic_dir

Semi-Supervised-MDD/scripts/timit_data_prep.sh $timit_dir "60-40" || exit 1;

python Semi-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_train  
python Semi-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_dev  
python Semi-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_test
mv ${data_dir}/l2_dev ${data_dir}/dev  
mv ${data_dir}/l2_test ${data_dir}/test
Semi-Supervised-MDD/scripts/timit_l2_merge.sh ${data_dir}/train_timit ${data_dir}/l2_train ${data_dir}/train
python Semi-Supervised-MDD/scripts/trans_prep_g2p.py --l2arctic_dir=$l2_arctic_dir\
    --timit_dir=$timit_dir --save_path=$data_dir

rm -rf l2_train train_timit

python Semi-Supervised-MDD/scripts/get_model_units.py $data_dir/train/phn_text $data_dir/label_units
python Semi-Supervised-MDD/scripts/get_model_units.py $data_dir/train/trans_g2p $data_dir/trans_units

# prepare manifest files
python Semi-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
    --dest $data_dir\
    --segment train\
    --scp_path $data_dir/train/wav.scp

python Semi-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
    --dest $data_dir\
    --segment valid\
    --scp_path $data_dir/dev/wav.scp

python Semi-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
    --dest $data_dir\
    --segment test\
    --scp_path $data_dir/test/wav.scp

python Semi-Supervised-MDD/scripts/generate_manifest_unlabeled.py $all_data_dir\
    --dest $data_dir\
    --l2_path $l2_arctic_dir

# prepare labels
python Semi-Supervised-MDD/scripts/generate_labels.py\
    --dest $data_dir\
    --segment train\
    --phn_text_path $data_dir/train/phn_text

python Semi-Supervised-MDD/scripts/generate_labels.py\
    --dest $data_dir\
    --segment valid\
    --phn_text_path $data_dir/dev/phn_text

python Semi-Supervised-MDD/scripts/generate_labels.py\
    --dest $data_dir\
    --segment test\
    --phn_text_path $data_dir/test/phn_text

python Semi-Supervised-MDD/scripts/generate_dict.py --data_dir $data_dir
```

以上脚本中的几个参数释义如下：

- timit_dir：TIMIT数据集的路径
- raw_l2_arctic_dir：L2-ARCTIC v5.0数据集的路径
- l2_arctic_dir：脚本输出L2-ARCTIC降采样到16k音频文件的路径
- all_data_dir：timit_dir和l2_arctic_dir所在的路径，它们应该放在同一路径下
- data_dir：脚本输出数据文件的路径

另外，需要克隆[G2P工具](https://github.com/petronny/g2p)到 Semi-Supervised-MDD/scripts/ 路径下，并进行安装。

## 模型训练

```shell
# 在TIMIT和L2-Arctic上微调Wav2Vec2.0预训练模型，wav2vec_small.pt从https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt下载
HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py\
    --config-dir ./Semi-Supervised-MDD/config\
    --config-name finetune_base_t\
    common.tensorboard_logdir=$output_dir/finetune_base_t\
    task.data=$data_dir\
    model.w2v_path=wav2vec_small.pt\
    hydra.run.dir=$output_dir/$finetune_base_t\
    checkpoint.save_dir=$output_dir/$finetune_base_t

# 在TIMIT和L2-Arctic上进行半监督学习
HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py\
    --config-dir ./Semi-Supervised-MDD/config\
    --config-name mean_teacher_ft_sample_focal_ctc_gamma4\
    common.tensorboard_logdir=$output_dir/mean_teacher_ft_sample_focal_ctc_gamma4\
    task.data=$data_dir\
    task.trans_path=$data_dir/train_u/trans_g2p\
    model.w2v_path=$output_dir/finetune_base_t/checkpoint_best.pt\
    hydra.run.dir=$output_dir/$mean_teacher_ft_sample_focal_ctc_gamma4\
    checkpoint.save_dir=$output_dir/$mean_teacher_ft_sample_focal_ctc_gamma4
```

## 模型推理

```shell
python ../Semi-Supervised-MDD/scripts/decode.py\
      --checkpoint_path $output_dir/mean_teacher_ft_sample_focal_ctc_gamma4/checkpoint_best.pt\
      --config_name $output_dir/mean_teacher_ft_sample_focal_ctc_gamma4/\
      --data_dir $data_dir\
      --segment test\
      --output_dir $output_dir
```

## MDD结果分析

```shell
. ./path.sh
./Semi-Supervised-MDD/scripts/mdd_result.sh $mean_teacher_ft_sample_focal_ctc_gamma4 $data_dir test $output_dir
cat $output_dir/mean_teacher_ft_sample_focal_ctc_gamma4/mdd_result
```
