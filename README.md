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
##### Small Model (100 Hidden Layers)
[CPU Simple Small Model Full Logs](/logs/cpu_simple_small.txt)
``````
poch: 491  time/epoch: 0.117  correct: 49 loss: 1.8380109654322383
Epoch: 492  time/epoch: 0.102  correct: 48 loss: 0.18852553004727557
Epoch: 493  time/epoch: 0.1  correct: 49 loss: 0.43735208086087113
Epoch: 494  time/epoch: 0.101  correct: 49 loss: 2.0487971108145366
Epoch: 495  time/epoch: 0.098  correct: 49 loss: 0.3313782260199691
Epoch: 496  time/epoch: 0.103  correct: 49 loss: 1.1217636602836873
Epoch: 497  time/epoch: 0.101  correct: 49 loss: 0.9122869560250999
Epoch: 498  time/epoch: 0.115  correct: 50 loss: 0.3245628453834277
Epoch: 499  time/epoch: 0.1  correct: 50 loss: 0.4232955881340918
``````

##### Large Model (200 Hidden Layers)
[CPU Simple Large Model Full Logs](/logs/cpu_simple_large.txt)
```
Epoch: 491  time/epoch: 0.396  correct: 50 loss: 0.2209945824060412
Epoch: 492  time/epoch: 0.411  correct: 50 loss: 0.6070185227374958
Epoch: 493  time/epoch: 0.4  correct: 50 loss: 0.645320656407705
Epoch: 494  time/epoch: 0.414  correct: 50 loss: 0.5605561495589545
Epoch: 495  time/epoch: 0.372  correct: 50 loss: 0.4620321894051821
Epoch: 496  time/epoch: 0.4  correct: 49 loss: 0.0245559022701354
Epoch: 497  time/epoch: 0.408  correct: 47 loss: 1.1987126856587431
Epoch: 498  time/epoch: 0.426  correct: 50 loss: 0.4819944499241963
Epoch: 499  time/epoch: 0.42  correct: 50 loss: 0.15313880909923205

```

#### Dataset 2: Split
##### Small Model (100 Hidden Layers)
[CPU Split Small Model Full Logs](/logs/cpu_split_small.txt)

```
Epoch: 491  time/epoch: 0.102  correct: 50 loss: 0.8172930346140115
Epoch: 492  time/epoch: 0.099  correct: 48 loss: 0.9858267357486696
Epoch: 493  time/epoch: 0.114  correct: 50 loss: 0.07012539075801523
Epoch: 494  time/epoch: 0.105  correct: 48 loss: 0.1989689883889654
Epoch: 495  time/epoch: 0.098  correct: 50 loss: 0.09261635327440519
Epoch: 496  time/epoch: 0.103  correct: 49 loss: 0.9515715172991421
Epoch: 497  time/epoch: 0.102  correct: 50 loss: 0.06900837839522407
Epoch: 498  time/epoch: 0.099  correct: 49 loss: 0.9850996613194296
Epoch: 499  time/epoch: 0.102  correct: 50 loss: 0.21443643562367418
```

##### Large Model (200 Hidden Layers)
[CPU Split Large Model Full Logs](/logs/cpu_split_large.txt)
```
Epoch: 491  time/epoch: 0.396  correct: 50 loss: 0.2209945824060412
Epoch: 492  time/epoch: 0.411  correct: 50 loss: 0.6070185227374958
Epoch: 493  time/epoch: 0.4  correct: 50 loss: 0.645320656407705
Epoch: 494  time/epoch: 0.414  correct: 50 loss: 0.5605561495589545
Epoch: 495  time/epoch: 0.372  correct: 50 loss: 0.4620321894051821
Epoch: 496  time/epoch: 0.4  correct: 49 loss: 0.0245559022701354
Epoch: 497  time/epoch: 0.408  correct: 47 loss: 1.1987126856587431
Epoch: 498  time/epoch: 0.426  correct: 50 loss: 0.4819944499241963
Epoch: 499  time/epoch: 0.42  correct: 50 loss: 0.15313880909923205

```


### GPU Logs