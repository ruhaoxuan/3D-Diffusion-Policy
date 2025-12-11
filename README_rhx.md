# dp3 数据处理

## 注意点

1. 由于分辨率太高内存炸了，看到 dp3 的实验用的是 84*84 的图像，现在在 generate_zarr.py 中手动调整了分辨率，可以看看有没有办法在模拟器输出时就输出 84 * 84 的图像。
2. 现在是通过上2帧的 ee_state + gripper_control 作为 state ，去预测下8帧的这两个数值作为 action。
3. 相机内参问题不确定对不对



## 运行

1. 生成数据：extra_rhx 目录下运行：

```bash
conda activate dp3
python src/generate_zarr.py --data_dir ./data/banana_plate/ --save_dir ../3D-Diffusion-Policy/data/ --env_name pick
```



2. 训练：参考 train_policy.sh，在仓库根目录（3D-Diffusion-Policy）下输入：

```
bash scripts/train_policy.sh dp3 pick 0112 0 0
```



## 输入格式

使用 observation 的 dict

```python
obs_dict = {
	'agent_pos': shape(2, 8),
	'point_cloud': shape(2, 512, 3),
}

return action: shape(8, 8)
```



obs 的 T 维度可以更长，会裁剪。输入只需要是 numpy 数组。

在  3D-Diffusion-Policy 下运行：

```bash
bash scripts/inference_policy.sh dp3 pick 0112 0 0
```



目前通过 json 进行输入输出，下一步将其接入到 ros 系统上。