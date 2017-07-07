## CUDA fun
I watched an Nvidia webinar and was blown away that GPUs had literally thousands of cores. I had some CUDA programming experience from my Operating Systems class at NC State, but we "only" had a few hundred cores to work with. I decided to learn more about CUDA programming so I could get more experienced at taking advantage of massive parallelism.
## Hardware
I rented a EC2 P2 instance from AWS, which gave me access to a Tesla K80 - a $4,000 GPU.
## My process
Currently, I've only implemented matrix multiplication. However, that process has taught me a lot.
I made a single-threaded CPU implementation and a multi-threaded CPU implemented. Then, I tried my hand at writing a CUDA implementation.
My first attempt ran slower than the multi-threaded CPU implementation. Changing around the kernel only helped so much.
After using shared memory and cleverly blocking up the result matrix, the final CUDA implementation was about 35 times as fast as the multi-threaded CPU implementation.
## Folder explanations
The root folder contains the single-threaded CPU implementation.
The CPU_threaded folder contains the CPU threaded implementation.
The GPU_threaded_1 folder contains my first attempt at a CUDA solution (which didn't work out too well).
The GPU_threaded_2 folder contains my second attempt at a CUDA solution (which wasn't much better).
The GPU_threaded_3 folder contains my best CUDA solution, which is about 35 times faster than the CPU multi-threaded implementation.
## License
MIT License

Copyright (c) 2017 Lucas Molander

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
