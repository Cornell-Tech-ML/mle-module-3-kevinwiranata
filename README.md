[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)
# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py


## Parallel Check Script Output
```
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (154)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (154)
-------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                                        |
        out: Storage,                                                                                                                |
        out_shape: Shape,                                                                                                            |
        out_strides: Strides,                                                                                                        |
        in_storage: Storage,                                                                                                         |
        in_shape: Shape,                                                                                                             |
        in_strides: Strides,                                                                                                         |
    ) -> None:                                                                                                                       |
        if (np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape) and len(in_shape) == len(out_shape)):    |
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                                                           |
        else:                                                                                                                        |
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------| #1
                out_index = np.empty(len(out_shape), np.int32)                                                                       |
                in_index = np.empty(len(in_shape), np.int32)                                                                         |
                to_index(i, out_shape, out_index)                                                                                    |
                broadcast_index(out_index, out_shape, in_shape, in_index)                                                            |
                out[index_to_position(out_index, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (167) is hoisted out of the
parallel loop labelled #1 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (168) is hoisted out of the
parallel loop labelled #1 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (197)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (197)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                          |
        out: Storage,                                                                                                                                                  |
        out_shape: Shape,                                                                                                                                              |
        out_strides: Strides,                                                                                                                                          |
        a_storage: Storage,                                                                                                                                            |
        a_shape: Shape,                                                                                                                                                |
        a_strides: Strides,                                                                                                                                            |
        b_storage: Storage,                                                                                                                                            |
        b_shape: Shape,                                                                                                                                                |
        b_strides: Strides,                                                                                                                                            |
    ) -> None:                                                                                                                                                         |
        if (len(a_shape) == len(b_shape) and np.array_equal(a_strides, b_strides) and np.array_equal(a_shape, b_shape) and np.array_equal(b_strides, out_strides)):    |
            for i in prange(len(out)):---------------------------------------------------------------------------------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                |
        else:                                                                                                                                                          |
            for i in prange(len(out)):---------------------------------------------------------------------------------------------------------------------------------| #3
                a_index = np.empty(len(a_shape), np.int32)                                                                                                             |
                b_index = np.empty(len(b_shape), np.int32)                                                                                                             |
                out_index = np.empty(len(out_shape), np.int32)                                                                                                         |
                to_index(i, out_shape, out_index)                                                                                                                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                |
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                |
                a_pos = index_to_position(a_index, a_strides)                                                                                                          |
                b_pos = index_to_position(b_index, b_strides)                                                                                                          |
                out_pos = index_to_position(out_index, out_strides)                                                                                                    |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (213) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (214) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (215) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (245)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (245)
-----------------------------------------------------------------------------|loop #ID
    def _reduce(                                                             |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        a_storage: Storage,                                                  |
        a_shape: Shape,                                                      |
        a_strides: Strides,                                                  |
        reduce_dim: int,                                                     |
    ) -> None:                                                               |
        # loop through everything fisrt                                      |
        for i in prange(len(out)):-------------------------------------------| #5
            out_index: Index = np.zeros(MAX_DIMS, np.int32)------------------| #4
            a_index = out_index                                              |
            to_index(i, out_shape, out_index)                                |
            out_pos = index_to_position(out_index, out_strides)              |
            # loop through dimension to reduce and do the function thingy    |
            for j in range(a_shape[reduce_dim]):                             |
                a_index[reduce_dim] = j                                      |
                a_pos = index_to_position(a_index, a_strides)                |
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #5, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--5 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (serial)



Parallel region 0 (loop #5) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mle-
module-3-kevinwiranata/minitorch/fast_ops.py (256) is hoisted out of the
parallel loop labelled #5 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (269)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mle-module-3-kevinwiranata/minitorch/fast_ops.py (269)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """                                                                                   |
    NUMBA tensor matrix multiply function.                                                |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
        None : Fills in `out`                                                             |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    for x in prange(out_shape[0]):--------------------------------------------------------| #8
        for y in prange(out_shape[1]):----------------------------------------------------| #7
            for z in prange(out_shape[2]):------------------------------------------------| #6
                res = 0.0                                                                 |
                posA = x * a_batch_stride + y * a_strides[1]                              |
                posB = x * b_batch_stride + z * b_strides[2]                              |
                for _ in range(b_shape[1]):  # b_shape[1] == a_shape[2]                   |
                    res += a_storage[posA] * b_storage[posB]  # dot product               |
                    posA += a_strides[2]  # col                                           |
                    posB += b_strides[1]  # row                                           |
                out_pos = x * out_strides[0] + y * out_strides[1] + z * out_strides[2]    |
                out[out_pos] = res                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
      +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)
      +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)
      +--6 (serial)



Parallel region 0 (loop #8) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Training Output Logs
**CPU** = Apple M1 Pro 2021 (Sonoma 14.1) <br>
**GPU** = Google Colab T4 GPU
### CPU Logs

#### Dataset 1: Simple

##### Small Model (40 Hidden Layers)
[CPU Simple Small Model Full Logs](/logs/cpu_simple_hidden_40.txt)
``````
Epoch: 490  time/epoch: 0.039  correct: 50 loss: 9.35136396147478e-05
Epoch: 491  time/epoch: 0.039  correct: 50 loss: 0.0041865385105562
Epoch: 492  time/epoch: 0.038  correct: 50 loss: 0.006429745821012849
Epoch: 493  time/epoch: 0.039  correct: 50 loss: 0.17032792178675937
Epoch: 494  time/epoch: 0.038  correct: 50 loss: 0.0326354427327606
Epoch: 495  time/epoch: 0.039  correct: 50 loss: 0.39587471246486666
Epoch: 496  time/epoch: 0.04  correct: 50 loss: 0.13469897805408265
Epoch: 497  time/epoch: 0.04  correct: 50 loss: 0.6213224608633433
Epoch: 498  time/epoch: 0.042  correct: 50 loss: 0.12548621566815693
Epoch: 499  time/epoch: 0.042  correct: 50 loss: 0.018896435750436577
``````

##### Large Model (200 Hidden Layers)
[CPU Simple Large Model Full Logs](/logs/cpu_simple_hidden_200.txt)
```
Epoch: 490  time/epoch: 0.108  correct: 50 loss: 0.09784735623814257
Epoch: 491  time/epoch: 0.111  correct: 50 loss: 0.0011947367809209458
Epoch: 492  time/epoch: 0.111  correct: 50 loss: 0.031182582738508095
Epoch: 493  time/epoch: 0.11  correct: 50 loss: 0.10180049908004778
Epoch: 494  time/epoch: 0.111  correct: 50 loss: 0.033434525517599775
Epoch: 495  time/epoch: 0.11  correct: 50 loss: 0.022194202381972665
Epoch: 496  time/epoch: 0.108  correct: 50 loss: 0.058828618403537244
Epoch: 497  time/epoch: 0.106  correct: 50 loss: 0.005990732330378058
Epoch: 498  time/epoch: 0.111  correct: 50 loss: 0.001638181379733589
Epoch: 499  time/epoch: 0.137  correct: 50 loss: 0.24869153320376697
```

#### Dataset 2: Split
##### Small Model (40 Hidden Layers)
[CPU Split Small Model Full Logs](/logs/cpu_split_hidden_40.txt)

```
Epoch: 490  time/epoch: 0.055  correct: 49 loss: 1.48960119515632
Epoch: 491  time/epoch: 0.05  correct: 49 loss: 0.5767885028951708
Epoch: 492  time/epoch: 0.052  correct: 49 loss: 0.2420677393429998
Epoch: 493  time/epoch: 0.041  correct: 49 loss: 0.16160638812859843
Epoch: 494  time/epoch: 0.041  correct: 49 loss: 0.34227825097964176
Epoch: 495  time/epoch: 0.047  correct: 49 loss: 1.1109076001726468
Epoch: 496  time/epoch: 0.04  correct: 50 loss: 0.1795246980579488
Epoch: 497  time/epoch: 0.044  correct: 49 loss: 0.7038291749059226
Epoch: 498  time/epoch: 0.039  correct: 49 loss: 0.1262845442416151
Epoch: 499  time/epoch: 0.041  correct: 50 loss: 0.2388534171875122
```

##### Large Model (200 Hidden Layers)
[CPU Split Large Model Full Logs](/logs/cpu_split_hidden_200.txt)
```
Epoch: 490  time/epoch: 0.104  correct: 50 loss: 0.18963312177357697
Epoch: 491  time/epoch: 0.104  correct: 50 loss: 0.23899689371147
Epoch: 492  time/epoch: 0.105  correct: 50 loss: 0.025903794158032085
Epoch: 493  time/epoch: 0.108  correct: 50 loss: 0.14221251265388776
Epoch: 494  time/epoch: 0.107  correct: 50 loss: 0.08181385391653838
Epoch: 495  time/epoch: 0.107  correct: 50 loss: 0.22122123099039764
Epoch: 496  time/epoch: 0.106  correct: 50 loss: 0.20162109659809796
Epoch: 497  time/epoch: 0.107  correct: 50 loss: 0.3243252551023929
Epoch: 498  time/epoch: 0.103  correct: 50 loss: 0.03586016343753507
Epoch: 499  time/epoch: 0.105  correct: 50 loss: 0.0632512405095207
```

#### Dataset 3: XOR
##### Small Model (40 Hidden Layers)
[CPU XOR Small Model Full Logs](/logs/cpu_xor_hidden_40.txt)

```
Epoch: 490  time/epoch: 0.048  correct: 50 loss: 0.353932476131671
Epoch: 491  time/epoch: 0.043  correct: 50 loss: 0.4346249030450587
Epoch: 492  time/epoch: 0.044  correct: 50 loss: 0.20982407570994352
Epoch: 493  time/epoch: 0.041  correct: 50 loss: 0.5670210685055194
Epoch: 494  time/epoch: 0.041  correct: 50 loss: 0.08192992909261898
Epoch: 495  time/epoch: 0.044  correct: 50 loss: 0.7079373795372923
Epoch: 496  time/epoch: 0.04  correct: 50 loss: 0.3324921233251884
Epoch: 497  time/epoch: 0.048  correct: 50 loss: 0.2515805944338293
Epoch: 498  time/epoch: 0.052  correct: 50 loss: 0.587985910569525
Epoch: 499  time/epoch: 0.057  correct: 50 loss: 0.24512795711797666
```

##### Large Model (200 Hidden Layers)
[CPU XOR Large Model Full Logs](/logs/cpu_xor_hidden_200.txt)
```
Epoch: 490  time/epoch: 0.122  correct: 50 loss: 0.5499636749810255
Epoch: 491  time/epoch: 0.111  correct: 50 loss: 0.4412281049084891
Epoch: 492  time/epoch: 0.109  correct: 50 loss: 0.3001558608446644
Epoch: 493  time/epoch: 0.107  correct: 50 loss: 0.5466758546715463
Epoch: 494  time/epoch: 0.109  correct: 50 loss: 0.9098846998562004
Epoch: 495  time/epoch: 0.106  correct: 49 loss: 0.4579630306186758
Epoch: 496  time/epoch: 0.108  correct: 50 loss: 0.36159400644326195
Epoch: 497  time/epoch: 0.109  correct: 50 loss: 0.3004750894542475
Epoch: 498  time/epoch: 0.113  correct: 50 loss: 0.06292743283668693
Epoch: 499  time/epoch: 0.109  correct: 50 loss: 0.36189260097331766
```

### GPU Logs

#### Dataset 1: Simple
##### Small Model (40 Hidden Layers)
[GPU Simple Small Model Full Logs](/logs/gpu_simple_hidden_40.txt)
``````
Epoch: 490  time/epoch: 2.258  correct: 50 loss: 0.669243913269936
Epoch: 491  time/epoch: 2.635  correct: 47 loss: 0.012752511048284948
Epoch: 492  time/epoch: 1.858  correct: 50 loss: 0.4773855286498626
Epoch: 493  time/epoch: 1.969  correct: 50 loss: 0.3179427853413576
Epoch: 494  time/epoch: 1.869  correct: 49 loss: 0.8192009528365494
Epoch: 495  time/epoch: 1.916  correct: 50 loss: 0.9707794059629369
Epoch: 496  time/epoch: 2.021  correct: 49 loss: 0.26315440897402803
Epoch: 497  time/epoch: 2.217  correct: 49 loss: 0.8223024843602151
Epoch: 498  time/epoch: 1.834  correct: 50 loss: 0.3544859270705081
Epoch: 499  time/epoch: 1.923  correct: 50 loss: 0.1382104278653334
``````

##### Large Model (200 Hidden Layers)
[GPU Simple Large Model Full Logs](/logs/gpu_simple_hidden_200.txt)
```
Epoch: 490  time/epoch: 2.649  correct: 50 loss: 0.0898665854453213
Epoch: 491  time/epoch: 2.186  correct: 50 loss: 0.010849949155493753
Epoch: 492  time/epoch: 2.444  correct: 50 loss: 0.0001535534359212782
Epoch: 493  time/epoch: 2.496  correct: 50 loss: 0.0016656000439290778
Epoch: 494  time/epoch: 2.537  correct: 50 loss: 0.0213044954611823
Epoch: 495  time/epoch: 1.997  correct: 50 loss: 0.0366721437665844
Epoch: 496  time/epoch: 1.99  correct: 50 loss: 0.0041036867730939565
Epoch: 497  time/epoch: 2.054  correct: 50 loss: 0.023411332145378463
Epoch: 498  time/epoch: 1.996  correct: 50 loss: 0.05201037043627177
Epoch: 499  time/epoch: 2.143  correct: 50 loss: 0.10436034954534351
```

#### Dataset 2: Split
##### Small Model (40 Hidden Layers)
[GPU Split Small Model Full Logs](/logs/gpu_split_hidden_40.txt)
``````
Epoch: 490  time/epoch: 1.891  correct: 50 loss: 0.2925929603951757
Epoch: 491  time/epoch: 1.862  correct: 50 loss: 0.2437077484149956
Epoch: 492  time/epoch: 1.888  correct: 50 loss: 0.26714149507198537
Epoch: 493  time/epoch: 2.377  correct: 50 loss: 0.22061432385473745
Epoch: 494  time/epoch: 1.88  correct: 50 loss: 0.4241978701407801
Epoch: 495  time/epoch: 2.234  correct: 50 loss: 0.20624496627865826
Epoch: 496  time/epoch: 2.309  correct: 50 loss: 0.10114756412748278
Epoch: 497  time/epoch: 1.975  correct: 50 loss: 0.42677082415403933
Epoch: 498  time/epoch: 1.854  correct: 50 loss: 0.20135839673846015
Epoch: 499  time/epoch: 1.845  correct: 50 loss: 0.28199865131379154
``````

[GPU Split Large Model Full Logs](/logs/gpu_split_hidden_200.txt)
``````
Epoch: 490  time/epoch: 1.994  correct: 50 loss: 0.0460216830815995
Epoch: 491  time/epoch: 2.29  correct: 50 loss: 0.17172398291768307
Epoch: 492  time/epoch: 2.495  correct: 50 loss: 0.006852126788436738
Epoch: 493  time/epoch: 2.461  correct: 50 loss: 0.07905437550800748
Epoch: 494  time/epoch: 2.03  correct: 50 loss: 0.006275140148132662
Epoch: 495  time/epoch: 1.946  correct: 50 loss: 0.01641350003145869
Epoch: 496  time/epoch: 1.961  correct: 50 loss: 0.04182278020659123
Epoch: 497  time/epoch: 2.017  correct: 50 loss: 0.16695578691649957
Epoch: 498  time/epoch: 2.006  correct: 50 loss: 0.0789306618295536
Epoch: 499  time/epoch: 1.955  correct: 50 loss: 0.031103915554165702
``````

#### Dataset 3: XOR
##### Small Model (40 Hidden Layers)
[GPU XOR Small Model Full Logs](/logs/gpu_xor_hidden_40.txt)
``````
Epoch: 490  time/epoch: 3.153  correct: 48 loss: 0.6854577289726315
Epoch: 491  time/epoch: 3.192  correct: 48 loss: 0.9608469681394641
Epoch: 492  time/epoch: 3.227  correct: 48 loss: 1.407126798488358
Epoch: 493  time/epoch: 2.424  correct: 48 loss: 1.2213035793072395
Epoch: 494  time/epoch: 1.865  correct: 48 loss: 0.7307279646732243
Epoch: 495  time/epoch: 1.884  correct: 48 loss: 0.38702393484002423
Epoch: 496  time/epoch: 1.869  correct: 48 loss: 2.0102763513713238
Epoch: 497  time/epoch: 1.912  correct: 49 loss: 0.5965750579864846
Epoch: 498  time/epoch: 2.355  correct: 49 loss: 1.6328201795392707
Epoch: 499  time/epoch: 2.884  correct: 49 loss: 1.4304389312716437
``````

[GPU XOR Large Model Full Logs](/logs/gpu_xor_hidden_200.txt)
``````
Epoch: 490  time/epoch: 2.121  correct: 50 loss: 0.07158161310515006
Epoch: 491  time/epoch: 2.457  correct: 50 loss: 0.06938989938088509
Epoch: 492  time/epoch: 2.42  correct: 50 loss: 0.22446716220895815
Epoch: 493  time/epoch: 2.151  correct: 50 loss: 0.06363693661777009
Epoch: 494  time/epoch: 1.973  correct: 50 loss: 0.04353911295999932
Epoch: 495  time/epoch: 1.964  correct: 50 loss: 0.08008182948310376
Epoch: 496  time/epoch: 1.963  correct: 50 loss: 0.0835195617911093
Epoch: 497  time/epoch: 2.066  correct: 50 loss: 0.027788276858056394
Epoch: 498  time/epoch: 2.384  correct: 50 loss: 0.10232473587463337
Epoch: 499  time/epoch: 2.077  correct: 50 loss: 0.03639259978485761
``````
