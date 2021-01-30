# -------- Get a cloned version of Cirq --------

git clone git@github.com:tonybruguier/Cirq.git cirq_master
cd cirq_master
git remote add upstream git@github.com:quantumlib/Cirq.git
git pull upstream master

export PYTHONPATH="/home/ubuntu/Dropbox/AWS_EC2_shared/git/cirq_master"

# -------- install --------
for ii in `cat requirements.txt | grep -v cirq`
do
  python3 -m pip install $ii --no-cache-dir
done

python3 -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'

python3 -m pip install -U keras_applications keras_preprocessing

# -------- bazel --------
sudo apt-get remove bazel
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel_3.1.0-linux-x86_64.deb
sudo dpkg -i bazel_3.1.0-linux-x86_64.deb
sudo apt-mark hold bazel
bazel --version

# -------- Create new client --------
git_new_branch() {
    NAME_OF_MY_CLIENT=$1

    git clone git@github.com:tonybruguier/quantum.git ${NAME_OF_MY_CLIENT}

    cd ${NAME_OF_MY_CLIENT}

    git remote add upstream git@github.com:tensorflow/quantum.git
    git pull upstream master
    git branch ${NAME_OF_MY_CLIENT}
    git checkout ${NAME_OF_MY_CLIENT}
}

./configure
./scripts/test_all.sh

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
    $2
}
