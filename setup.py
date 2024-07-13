# Available at setup time due to pyproject.toml
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import pybind11
import numpy as np
from distutils.sysconfig import get_python_inc


class BuildExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler
        if isinstance(compiler, str):
            compiler = self.compiler = self._finalize_compiler()

        for ext in self.extensions:
            if any(f.endswith(".cu") for f in ext.sources):
                ext.extra_compile_args = {
                    "gcc": ["-std=c++17"],
                    "nvcc": ["-std=c++17", "-arch=sm_35"],
                }
                ext.sources = self.cuda_compile(ext.sources, ext.include_dirs)

        build_ext.build_extensions(self)

    def cuda_compile(self, sources, include_dirs):
        include_str = " ".join(f"-I{dir}" for dir in include_dirs)
        for source in sources:
            if source.endswith(".cu"):
                base, ext = os.path.splitext(source)
                target = base + ".o"
                cmd = f"nvcc -c {source} -o {target} -std=c++17 {include_str}"
                if os.system(cmd) != 0:
                    raise RuntimeError("nvcc compilation failed")
                sources[sources.index(source)] = target
        return sources


__version__ = "0.0.1"

ext_modules = [
    Extension(
        "scaling_democracy",
        ["scaling_democracy.cu"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),  # Include Python headers
        ],
        language="c++",
        extra_compile_args={
            "gcc": ["-std=c++17"],
            "nvcc": ["-std=c++17", "-arch=sm_35"],
        },
    ),
]

setup(
    name="scaling-democracy",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/scaling-democracy",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
