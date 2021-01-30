# -------- install --------
sudo apt install python-pytest
sudo apt install python3-pip

python3 -m pip install numpy
python3 -m pip install pytest
python3 -m pip install scipy
python3 -m pip install yapf
python3 -m pip install mypy
python3 -m pip install pylint
python3 -m pip install matplotlib
python3 -m pip install pandas
python3 -m pip install networkx
python3 -m pip install google
python3 -m pip install google-api-python-client
python3 -m pip install grpcio
python3 -m pip install sympy

python3 -m pip install sklearn
python3 -m pip install seaborn
python3 -m pip install sphinx
python3 -m pip install networkx

python3 -m pip install quimb
python3 -m pip install opt_einsum
python3 -m pip install autoray

# -------- Create new client --------
git_new_branch() {
    NAME_OF_MY_CLIENT=$1

    git clone git@github.com:tonybruguier/Cirq.git ${NAME_OF_MY_CLIENT}

    cd ${NAME_OF_MY_CLIENT}

    git remote add upstream git@github.com:quantumlib/Cirq.git
    git pull upstream master
    git branch ${NAME_OF_MY_CLIENT}
    git checkout ${NAME_OF_MY_CLIENT}
}

# -------- Run test --------
python3 -m pytest examples/direct_fidelity_estimation_test.py

