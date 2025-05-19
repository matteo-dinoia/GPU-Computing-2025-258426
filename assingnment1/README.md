# Assignment 1

To compile and test the code use the manual method (as the automated method is experimental).

Firstly clone this repository over the cluster.

## Cpu

Compile using

```bash
make cpu
```

And then run using

```bash
sbatch cpu.sbatch <input-dataset>
```

## Gpu

Compile using

```bash
make gpu
```

And then run using

```bash
sbatch gpu-runner.sbatch <input-dataset>
```