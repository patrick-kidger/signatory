# Installation Guide on MacOS

## Download the source code

- Clone the GitHub repository
    ```
    git clone https://github.com/patrick-kidger/signatory.git
    ```
- The recommended location is obtained by activating your environment and executing
    ```
    python -m site --user-site
    ```
    It is possible, that it must be created first.

## Ensure corresponding `PyTorch` version

- Make sure you have the corresponding `pytorch` version installed. You can look up matching pairs [here](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html#older-versions).

## Prepare clang for installation

- The `setup.py` uses the option `-fopenmp` which is not supported on MAC by default.
- To fix this, install `lvmm` and `libomp` from brew. For this
    - Make sure `Homebrew` is installed
    - Run 
    ```
    brew install lvmm
    ```
    - and 
    ``` 
    brew install libomp
    ```
- Go to the location of the cloned repository and modify `setup.py`: Change line `36` to
    ```
        extra_compile_args.append('-Xpreprocessor -fopenmp -lomp')
    ```    
- Modify 
    ```
    {path-to-repository}/signatory/src/pytorchbind.cpp
    ```
    by removing lines `38-40`, i.e.
    ```
    // #ifndef _OPENMP
    //     #error OpenMP required
    // #endif
    ```

## Run the installation command
- Run 
    ```
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
    ```



