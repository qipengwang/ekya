# Ekya: Continuous Learning on the Edge
Ekya is a system which enables continuous learning on resource constrained devices.

![Ekya Architecture](https://i.imgur.com/ng1jLsS.png)

More details can be found in our NSDI 2022 paper available [here](https://nsdi22spring.usenix.hotcrp.com/doc/nsdi22spring-paper74.pdf). 

# Installation

1. First checkout Ray repository. Ekya requires first building a particular branch of Ray from source. Ekya uses commit `cf53b351471716e7bfa71d36368ebea9b0e219c5` (`Ray 0.9.0.dev0`) from the Ray repository.
`pip install ray` is not sufficient. 
```bash
git clone https://github.com/ray-project/ray/
cd ray
git checkout cf53b35
```
2. To build Ray, follow the [build instructions](https://docs.ray.io/en/master/development.html#building-ray-on-linux-macos-full) from the Ray repository.
```
sudo apt-get update
sudo apt-get install -y build-essential curl unzip psmisc

# Install Cython
pip install cython==0.29.0 pytest

# Install Bazel.
ray/ci/travis/install-bazel.sh
# (Windows users: please manually place Bazel in your PATH, and point
# BAZEL_SH to MSYS2's Bash: ``set BAZEL_SH=C:\Program Files\Git\bin\bash.exe``)

# Build the dashboard
# (requires Node.js, see https://nodejs.org/ for more information).
pushd ray/dashboard/client
npm install
npm run build
popd

# Install Ray.
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```
3. After installing ray, clone the Ekya repository and install Ekya.
```
git clone https://github.com/romilbhardwaj/ekya/
pip install -e . --verbose
``` 
4. Install [Nvidia Multiprocess Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html).
```
sudo apt-get update
sudo apt-get install nvidia-cuda-mps
```
5. Set your GPU to run in exclusive process mode and run Nvidia MPS daemon. This will require killing Xserver if it is running.
```
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```

# Running Ekya
1. An example script is provided to run Ekya with the cityscapes dataset.
 ```
./ekya/experiment_drivers/driver_multicity.sh
``` 