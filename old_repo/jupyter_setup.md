# Setting up a Jupyter platform for Toliman modeling

Ben's investigations use Jupyter, with various Python packages from [Astroconda](http://astroconda.readthedocs.io/en/latest/index.html) along with [Poppy](http://pythonhosted.org/poppy/). This proved to be hard to run from my Windows 7 laptop due to some packages only being available for Linux and OSX. I ended up running a Ubuntu VM using [VirtualBox](https://www.virtualbox.org). This document captures most of the important details for posterity.

## Ubuntu VM

Create a new 64-bit Linux flavour VM in VirtualBox with 10 GB storage. Download [Ubuntu Server ISO](https://www.ubuntu.com/download/server) (I used a 16.04 LTS image). Boot up and install with minimal packages. I used a default user of "ubuntu".

```bash
sudo apt-get update
sudo apt-get dist-upgrade
```


## Shared Folder

Rather than putting all data inside the VM, I used an external shared folder. From VirtualBox VM settings->Shared Folders add a share to the required shared folder path, select Auto-mount and Make Permanent options. Folder name was Toliman, which becomes `/media/sf_Toliman` within the VM.

From within the VM you need to support shared folders by installing the guest utils. Disable recommended packages to avoid pulling all of XWindows in.

```bash
sudo apt-get install --no-install-recommends virtualbox-guest-utils
```

To allow the `ubuntu` user to access the shared folder if needs to belong to the `vboxsf` group.

```bash
sudo usermod -a -G vboxsf ubuntu
```

Reboot the VM to ensure things work properly (group priveleges agen't applied until next login).

I then added a symlink to the shared directory

```bash
ln -s /media/sf_Toliman/ toliman
```

## SSH

The VirtualBox VM console is lame and clipboard sharing functionality is beyond my patience to configure. Plus direct forwarding to the Jupyter web server doesn't work easily. Use PuTTY instead. Inside the VM install an SSH server.

```bash
sudo apt-get install openssh-server
```

Allow SSH access from VirtualBox VM Settings->Network: 
  * Attached to NAT
  * Advanced->Port Forwarding add "Name: SSH, Protocol: TCP, Host Port: 22, Guest Port: 22" 

Don't forget to restart the VM to refresh settings. In PuTTY create a new session to localhost and add a tunnel for Jupyter (Connection->SSH->Tunnels) Source port: 8888, Destination: localhost:8888.

## Astroconda

Based on [Astroconda docs](http://astroconda.readthedocs.io/en/latest/getting_started.html#installing-conda-the-choice-is-yours) I installed `miniconda` and added required additional packages. NOTE: this installs the Python 3 version, breaking some Python 2 scripts (especially `print` statements). But we all just need to move on, or we'd still be using Fortran (shudder).

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
echo "export PATH=\$PATH:\$HOME/miniconda/bin" > .bash_profile # for later
export PATH=$PATH:$HOME/miniconda/bin # for now
conda config --add channels http://ssb.stsci.edu/astroconda
conda create -n astroconda stsci
```

### FITS data

Although the `pysynphot` package is installed with the rest of Astroconda, the data files are not included, so following the [pysynphot install docs](http://pysynphot.readthedocs.io/en/latest/#installation-and-setup) you need do download all the tarballs from  [ftp://archive.stsci.edu/pub/hst/pysynphot/] and unpack them all in one place where `pysynphot` can find them. In my case I put the common `cdbs` folder in my `toliman` shared folder. Then set the location in the env:

```bash
echo "export PYSYN_CDBS=~/toliman/cdbs/" >> .bash_profile # for later
export PYSYN_CDBS=~/toliman/cdbs/ # for now
```

### Poppy

`poppy` isn't part of Astroconda, but extra packages can be added using the [docs](https://conda.io/docs/user-guide/tasks/manage-pkgs.html#id2):

```bash
conda install --name astroconda poppy
```

### Jupyter

Time to get started:

```bash
source activate astroconda
jupyter notebook
```

Note that Jupyter runs in the foreground, but you can always log in with additional PuTTY consoles or Ctrl-Z/fg.
