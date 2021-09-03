# -------- Get a cloned version of Cirq --------

git clone git@github.com:tonybruguier/Cirq.git cirq_master
cd cirq_master
git remote add upstream git@github.com:quantumlib/Cirq.git
git pull upstream master

export PYTHONPATH="/home/ubuntu/Dropbox/AWS_EC2_shared/git/cirq_master"

# -------- install --------
for ii in `cat requirements.txt | grep -v cirq`
do
  python3 -m pip install $ii
done

python3 -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'

python3 -m pip install -U keras_applications keras_preprocessing

python3 -m pip install pydot graphviz pydot_ng pydotplus qutip

sudo apt-get install graphviz

# -------- bazel --------
cd /tmp
sudo apt-get remove bazel
wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel_3.7.2-linux-x86_64.deb
sudo dpkg -i bazel_3.7.2-linux-x86_64.deb
sudo apt-mark hold bazel
bazel --version

# -------- Create new client --------

git_new_branch() {
    NAME_OF_MY_CLIENT=$1
    FROM_BRANCH=$2

    git clone git@github.com:tonybruguier/quantum.git ${NAME_OF_MY_CLIENT}

    cd ${NAME_OF_MY_CLIENT}

    git remote add upstream git@github.com:tensorflow/quantum.git
    git checkout ${FROM_BRANCH}
    git pull upstream ${FROM_BRANCH}
    git branch ${NAME_OF_MY_CLIENT}
    git checkout ${NAME_OF_MY_CLIENT}
}

# Make sure you are using python3!

./configure.sh
./scripts/test_all.sh
./scripts/build_pip_package_test.sh
bazel test -c opt --experimental_repo_remote_exec --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" --notest_keep_going --test_output=errors //tensorflow_quantum/python/layers/high_level/...

# -------- bazeltest --------
./configure

blaze() {
  bazel $1 \
    -c opt \
    --experimental_repo_remote_exec \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    --cxxopt="-msse2" \
    --cxxopt="-msse3" \
    --cxxopt="-msse4" \
    --notest_keep_going \
    --test_output=errors \
    --action_env=PYTHONPATH=${PYTHONPATH} \
    --keep_going \
    $2
}

# -------- format --------
python3 -m yapf --style=google --in-place path/to/file.py

# -------- Serialized Protos --------
python3 -m pip install protobuf~=3.13.0
python3 -m pip install grpcio-tools~=1.26.0
python3 -m pip install mypy-protobuf==1.10



# -------- running test --------
GIT_BASE_DIR="/home/ubuntu/Dropbox/AWS_EC2_shared/git"
CIRQ_MASTER_PATHS=""
for cirq_comp in `ls -d ${GIT_BASE_DIR}/cirq_master/cirq-* | awk -F"/" '{print $8}'`
do
  CIRQ_MASTER_PATHS="${CIRQ_MASTER_PATHS}:${GIT_BASE_DIR}/cirq_bump_proto_version/${cirq_comp}"
done
export PYTHONPATH=`echo ${CIRQ_MASTER_PATHS} | cut -c2-`

blaze test --test_filter=SerializerTest.test_deserialize_projectorsum_simple0 //tensorflow_quantum/core/serialize:serializer_test
blaze test //tensorflow_quantum/core/serialize:serializer_test

blaze test //tensorflow_quantum/core/serialize:op_serializer_test

blaze test //tensorflow_quantum/core/ops:cirq_ops_test