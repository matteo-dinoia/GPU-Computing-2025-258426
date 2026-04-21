# Deliverable 2
## Description
The full details can be found in the [paper pdf](report/deliverable2.pdf).

As in the previous deliverable we had to use the COO format.
<img width="895" height="481" alt="image" src="https://github.com/user-attachments/assets/b751dc84-33d7-4ab7-ab7d-d020d5c492bb" />

Various optimization such as cache optimization, prefix sum, optimized prefix sum, 
and optimization of parameter, unrolling and some other were tested.

## Results
<img width="718" height="478" alt="image" src="https://github.com/user-attachments/assets/4389e426-3bbf-478e-8fd7-8c0a2102de44" />

The final result was that we could improve the final performance of over 8x the baseline implementation.

The final result on the test machine ().
<img width="723" height="312" alt="image" src="https://github.com/user-attachments/assets/f939c5af-b3d9-45bf-9b7c-ad4230d0becd" />


## Local compilation & execution
It can be manually run using the executable but it is easier to just use the helper script, as such:
```bash
# Go in the root of the second deliverable and run
# The <dataset-to-use> is a path (even relative) to any file *.mtx *.bmtx *.sbmtx
$ ./scripts/locale-run.sh <dataset-to-use>
```

## Remote compilation & execution (eg. cluster)
### Running the project directly from the cluster
Clone the repository on the cluster and follow the local compilation steps.
### Running from the local machine (useful for development)
Can use the remote-run script. This require some setup. Go to the file remote-run.sh
and edit as appropriate.
```bash
# change to your own if any is needed
SSH_FLAG="-i <insert-ssh-file-if-present>"
# change to own
REMOTE="<change-user-name>@<change-remote-name-if-needed>"
```

Then it can be run as such:
```bash
# Go in the root of the second deliverable and run
$ ./scripts/remote-run.sh <dataset-to-use>
```

## Matrix conversion
As some of the kernel assume that the COO is sorted, it may be useful, when testing multiple times with the same matrix, to convert the matrix representation to a one smaller and possibly sorted.

### Small matrixes
To convert a matrix to a compact representation we can use (which is when the matrix is not too big)
```bast
./scripts/convert-matrix.sh <path-matrix-mtx-format>
```

### Big matrixes
When the matrix is huge 1Gb+ it is better to use, which drammatically reduce time wasted on execution. 

This should be done with each huge matrix, else the job may timeout on the cluster.

Also notice that this WILL TAKE A LONG TIME (it is recommended to disable any power saving feature).
```bast
./scripts/convert-huge-matrix-to-sorted.sh <path-matrix-mtx-or-btmx-format>
```
The converted matrix may be larger than the original in this case
