git_new_branch() {
    NAME_OF_MY_CLIENT=$1
    FROM_BRANCH=$2

    git clone git@github.com:tonybruguier/ReCirq.git ${NAME_OF_MY_CLIENT}

    cd ${NAME_OF_MY_CLIENT}

    git remote add upstream git@github.com:quantumlib/ReCirq.git
    git checkout ${FROM_BRANCH}
    git pull upstream ${FROM_BRANCH}
    git branch ${NAME_OF_MY_CLIENT}
    git checkout ${NAME_OF_MY_CLIENT}
}

python3 -m pip install absl-py

python3 recirq/qml_lfe/learn_states_q.py --n=3 --n_paulis=1 --save_dir=/tmp
