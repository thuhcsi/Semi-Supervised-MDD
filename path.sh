local_kaldi_path=/mnt/d/projects/kaldi
docker_kaldi_path=~/opt/kaldi

[ -d $local_kaldi_path ] && export KALDI_ROOT=$local_kaldi_path
[ -d $docker_kaldi_path ] && export KALDI_ROOT=$docker_kaldi_path

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C