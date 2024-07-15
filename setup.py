import os
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc, get_python_lib

import pybind11
import numpy as np


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                self.build_cuda_extension(ext)
            else:
                super().build_extension(ext)

    def build_cuda_extension(self, ext):
        # Compile CUDA source files
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_cuda(source)

        # Compile non-CUDA source files
        objects = []
        for source in ext.sources:
            if not source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source], output_dir=self.build_temp, extra_postargs=["-fPIC"]
                )
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "scaling_democracy.o")],
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def compile_cuda(self, source):
        # Compile CUDA source file using nvcc
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        include_dirs = self.compiler.include_dirs + ext.include_dirs
        include_dirs = " ".join(f"-I{dir}" for dir in include_dirs)
        output_file = os.path.join(output_dir, "scaling_democracy.o")

        # Kepler: -arch=sm_30
        # Turing: -arch=sm_75
        # Ampere: -arch=sm_86
        # Ada: -arch=sm_89
        # Hopper: -arch=sm_90
        # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        cmd = f"nvcc -c {source} -o {output_file} -std=c++17 -arch=sm_86 -Xcompiler -fPIC {include_dirs}"
        os.system(cmd)


__version__ = "0.0.1"

# Get Python library path dynamically
python_lib_dir = get_python_lib(standard_lib=True)
python_lib_name = os.path.basename(python_lib_dir).replace(".so", "")

ext_modules = [
    Extension(
        "scaling_democracy",
        ["scaling_democracy.cu"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),
            "/usr/local/cuda/include",
            "/usr/include/cuda",
            "cccl/cub/",
            "cccl/libcudacxx/include",
            "cccl/thrust/",
        ],
        library_dirs=[
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/wsl/lib",
            python_lib_dir,
        ],
        libraries=[
            "cudart",
            "cuda",
            "cufft",
            "cublas",
            python_lib_name.replace(".a", ""),
        ],
        extra_link_args=[f"-Wl,-rpath,{python_lib_dir}"],
        language="c++",
    ),
]

setup(
    name="scaling-democracy",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/scaling-democracy",
    description="GPU-accelerated Schulze voting algorithm",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
