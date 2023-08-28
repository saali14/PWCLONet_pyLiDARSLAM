#!/bin/bash

# create subprocess to enanle exiting script without killing terminal
# when errors occurred. This happens because we're "sourcing" the script [source setup.sh] 
(
    # exit when error occurs
    set -e

    # show input commands
    set -v

    
    # ----------------------------------------------------------------------------------------------
    # /!\ You should change the next values according to your personal preferences

    # set _WITH_PREFIX to 1 if you want to create the conda environment using `--prefix` option
    _WITH_PREFIX=1
    # if you set _WITH_PREFIX to 1 please change the directory where the 
    # environment will be added. Otherwise, just keep it this way
    _ENV_PATH=/data/melamine/environments
    # set _INSTALL_MINICONDA to 1 if you don't have conda already installed
    _INSTALL_MINICONDA=0
    # ----------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------
    # /!\ If you keep the same architecture of git repository then this should work just fine
    CONDA_DIR=$HOME/miniconda3
    _PATH_TO_RELIDAR_SLAM=..
    _PATH_TO_PYLIDAR_SLAM=.
    _ENV_NAME=pylidar-slam
    # ----------------------------------------------------------------------------------------------


    _ENV_FILE_PATH=${_PATH_TO_PYLIDAR_SLAM}/env.yml



    # installing miniconda
    if [ $_INSTALL_MINICONDA -eq 1 ];
    then
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh
        chmod +x ~/miniconda.sh
        ~/miniconda.sh -b -p $CONDA_DIR
        rm ~/miniconda.sh

        # Put conda in path so we can use conda activate
        PATH=$CONDA_DIR/bin:$PATH

        # make conda activate command available from /bin/bash --login shells
        echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

        # make conda activate command available from /bin/bash --interative shells
        # intilialize shell on conda
        conda init bash

        conda config --set env_prompt '({name})'
    fi

    conda activate
    # creating conda environment
    if [ $_WITH_PREFIX -eq 1 ];
    then
        conda create -y --prefix ${_ENV_PATH}/${_ENV_NAME}
        conda env update --prefix ${_ENV_PATH}/${_ENV_NAME} --file ${_ENV_FILE_PATH} --prune
        conda activate ${_ENV_PATH}/${_ENV_NAME}
    else
        conda env create -y --name ${_ENV_NAME} --file ${_ENV_FILE_PATH}
        conda activate ${_ENV_NAME}
    fi
    conda clean --all --yes

    # Installing env requirements using pip install
    pip install --extra-index-url https://rospypi.github.io/simple/ rosbag
    pip install -r ${_PATH_TO_RELIDAR_SLAM}/pyviz3d/requirements.txt && pip install ${_PATH_TO_RELIDAR_SLAM}/pyviz3d
    pip install -U g2o-python
    pip install -r ${_PATH_TO_PYLIDAR_SLAM}/requirements.txt


    #cd $CONDA_PREFIX
    #mkdir -p ./etc/conda/activate.d
    #mkdir -p ./etc/conda/deactivate.d
    #touch ./etc/conda/activate.d/env_vars.sh
    #touch ./etc/conda/deactivate.d/env_vars.sh

    #echo "#!/bin/sh" > ./etc/conda/activate.d/env_vars.sh
    #echo "" >> ./etc/conda/activate.d/env_vars.sh
    #echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib" >> ./etc/conda/activate.d/env_vars.sh

    #echo "#!/bin/sh" > ./etc/conda/deactivate.d/env_vars.sh
    #echo "" >> ./etc/conda/deactivate.d/env_vars.sh
    #echo "#unset LD_LIBRARY_PATH" > ./etc/conda/deactivate.d/env_vars.sh
)