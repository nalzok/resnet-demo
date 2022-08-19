# Train a ResNet-18 on MNIST with a TPU v3-8 in 2 minutes

Install dependencies with the following command

```bash
PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html pipenv update
```

Then just run `make`

```
$ make
time pipenv run python3 \
        -m demo.main \
        --batch_size 512 \
        --epochs 8 \
        --learning_rate 1e-2
WARNING:absl:Initialized persistent compilation cache at jit_cache
===> Training
Epoch 1, train loss: 13045.0380859375
Epoch 2, train loss: 3688.81396484375
Epoch 3, train loss: 2729.858642578125
Epoch 4, train loss: 2172.668701171875
Epoch 5, train loss: 1663.1148681640625
Epoch 6, train loss: 1300.9534912109375
Epoch 7, train loss: 1165.4324951171875
Epoch 8, train loss: 940.5345458984375
===> Testing
Train accuracy: 0.9915000200271606
Test accuracy: 0.983299970626831
177.10user 20.63system 1:51.40elapsed 177%CPU (0avgtext+0avgdata 4593904maxresident)k
0inputs+55936outputs (2major+1683318minor)pagefaults 0swaps
```

Note: I used `initialize_cache('jit_cache')` to persist the JIT compilation cache to the disk.
The `jit_cache/` directory might take several GiB, but feel free to delete it at any time.
