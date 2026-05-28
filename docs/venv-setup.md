Virtual-env and custom packages¶

The EASI JupyterLab docker images are built with many python data science and machine learning libraries. However, sometimes it may be desirable to add or test an additional library in your JupyterLab session.

There are two ways to do this:

    Create a virtual environment
    Build, test and contribute a docker image. Contact the EASI Core team for details and support in doing this.

Create a virtual environment (venv)¶

EASI JupyterLab images run a python venv so using pip install --user <package> will fail, as it does for all venvs.

See also this tutorial for more background, https://janakiev.com/blog/jupyter-virtual-envs/.

While using venv it is recommended to keep track of the packages being installed using a dependency management tool such as pip , uv or poetry to assist in recreating the virtualenv in the future.

    Create a blank environment, called "myenv" in this example.

    MYENV=myenv
    PYVERSION=$(python3 --version | awk '{print tolower($1$2)}' | sed 's/\.[0-9]*$//')
    python -m venv ~/venvs/$MYENV

    Link to the default environment for the base packages. If the image and python version is updated the venv will need to be recreated with a proper realpath matching the current image python version. Otherwise the kernel will seem to activate but cells will not execute. The PYVERSION variable exported in this script takes care of keeping it in sync.

    realpath /env/lib/$PYVERSION/site-packages > ~/venvs/$MYENV/lib/$PYVERSION/site-packages/base_venv.pth

    Activate the new environment.

    source ~/venvs/$MYENV/bin/activate

    Optionally, upgrade pip and update or install any additional packages. These installs will be available in the new environment only.

    pip install --upgrade pip <package1> <package2>

    Register the new environment as a Jupyter kernel.

    python -m ipykernel install --user --name=$MYENV --display-name "Your New Environment Name"

    In a notebook, select the new environment kernel (it may take a moment to become available).
        In the JupyterLab menu: Select Kernel > Change Kernel....
        In the top-right of a notebook: Click Python 3 label.
    Deactivate the venv and return to the default JupyterLab environment.

    deactivate

    As required, switch into the new environment to pip install or upgrade whatever extras you need.

    source ~/venvs/$MYENV/bin/activate
    pip install <package>

List and delete kernels¶

List available kernels

jupyter kernelspec list

Delete the "
myenv" kernel

jupyter kernelspec uninstall myenv

Jupyter Lab Extensions¶

If you have pip installed a Jupyter Lab extensions in a virtual environment any javascript component wont be automatically used by Jupyter Lab as it cant 'see' it in the virtual environment. These components can be moved/copied so Jupyter can 'see' them. After copying them refresh the browser tab.

cp -r /home/jovyan/venvs/new_env/share/jupyter/labextensions/* /home/jovyan/.local/share/jupyter/labextensions

Create a virtual environment from within a Jupyter Notebook and distribute to Dask workers¶

It is possible perform the above operations from within a Jupyter Notebook using shell calls from the cells. This can be useful if you want to distribute the notebook with all of the virtual-env customisation available in it. Change python version in the realpath command according to image and current python version.

To do this first create two cells. The first contains the environment information, the second runs the commands as described above:

env_path = 'new_env'
jupyter_kernel_name = 'new'
package_list = ['gdal_dask_reproject']

and

! [ ! -d ~/venvs/$env_path ] && python -m venv ~/venvs/$env_path
! source  ~/venvs/$env_path/bin/activate && python -m ipykernel install --user --name=$env_path --display-name=$jupyter_kernel_name
! realpath /env/lib/python3.8/site-packages > ~/venvs/$env_path/lib/python3.8/site-packages/base_venv.pth
packages_string = ' '.join(package_list)
! source  ~/venvs/$env_path/bin/activate &&  pip install --quiet $packages_string

After running those cells you need to restart your notebook and select the newly created kernel before you continue.

If you also need those packages to be available to dask workers you can add a cell after you create a cluster and client but before you scale it:

from dask.distributed import PipInstall
plugin = PipInstall(packages=package_list)
client.register_worker_plugin(plugin)

Further information:

    https://distributed.dask.org/en/latest/plugins.html#built-in-worker-plugins

Distribute custom packages to Dask workers¶

It is also possible to distribute custom (non-pypi) python packages to Dask workers. This may be useful if you are developing a package or using a package developed elsewhere but which isn't on pypi.

Build the package into an egg:

# Change to the package directory, and assuming the package has a setup.py file
python3 setup.py bdist_egg

From a Jupyter notebook, copy the egg file from the dist directory to a running dask cluster's workers:

client.upload_file(my_egg_file)

The egg file should automatically be imported into the worker environment.

Further information:

    http://distributed.dask.org/en/stable/api.html#distributed.Client.upload_file
