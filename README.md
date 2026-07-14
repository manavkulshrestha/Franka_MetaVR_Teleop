Build and Install polymetis

```
conda activate franka311 && cd ~/fairo/polymetis/polymetis && \
rm -rf build && mkdir build && cd build && \
PATH="/usr/bin:$PATH" cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_FRANKA=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_DOCS=OFF \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBOOST_ROOT="$CONDA_PREFIX" \
  -DEIGEN3_INCLUDE_DIR="$CONDA_PREFIX/include/eigen3" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" && \
PATH="/usr/bin:$PATH" make -j"$(nproc)"

pip install omegaconf hydra-core grpcio
cd ~/fairo/polymetis/polymetis && pip install -e .

# For warnings
ln -sf ~/fairo/polymetis/polymetis/build/torch_isolation/libtorchscript_pinocchio.so \
  "$CONDA_PREFIX/lib/libtorchscript_pinocchio.so"

ln -sf ~/fairo/polymetis/polymetis/build/torch_isolation/libtorchrot.so \
  "$CONDA_PREFIX/lib/libtorchrot.so"
```

Run robot server
```
sudo pkill -9 run_server; launch_robot.py robot_client=fr3_hardware robot_model=fr3 use_real_time=true exec=franka_panda_client +robot_client._recursive_=false robot_client.executable_cfg.robot_client_metadata_path=/home/bera/fairo/polymetis/polymetis/conf/default_metadata.yaml
```

Run gripper server
```
sudo pkill -9 run_gripper_server; launch_gripper.py gripper=franka_hand +gripper._recursive_=false
```