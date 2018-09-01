## Python Imagenet data loader

Provides loading/format changes for imagenet dataset.

### Setup
1. Download data from [here](http://image-net.org/download).
2. Extract data to some directory. The expected structure is as follows:
```
* IMAGENET_ROOT
  * ILSVRC2012_validation_ground_truth.txt
  * meta.mat
  - tarred
    * ILSVRC2012_img_val.tar
    - ILSVRC2012_img_train
      * n01440764.tar
      * n01443537.tar
      ...
```
3. Get this repository and [dids](https://github.com/jackd/dids) dependency:
```
cd /path/to/parent_dir
git clone https://github.com/jackd/dids.git
git clone https://github.com/jackd/imagenet.git
```
4. Export `IMAGENET_PATH`:
```
export IMAGENET_PATH=/path/to/IMAGENET_ROOT
```
5. Add `parent_dir` to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```


### Using zipped data
If you'd prefer to use `zip` archives rather than `tar`, running
```
./scripts/tar_to_zip.py
```
will create the relevant data that can be accessed through `imagenet.zipped` functions. If you wish to delete the `.tar` files once an equivalent `.zip` has been created, use
```
./scripts/tar_to_zip.py --delete_after_copy=True
```
