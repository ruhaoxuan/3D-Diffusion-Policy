# Examples:
# bash scripts/train_policy.sh -a dp3 -t adroit_hammer -i 0322 -s 0 -g 0
# bash scripts/train_policy.sh -a dp3 -t dexart_laptop -i 0322 -s 0 -g 0
# bash scripts/train_policy.sh -a simple_dp3 -t adroit_hammer -i 0322 -s 0 -g 0
# bash scripts/train_policy.sh -a dp3 -t metaworld_basketball -i 0602 -s 0 -g 0



DEBUG=False
save_ckpt=True
seed=0
gpu_id=0
alg_name="dp3"
addition_info="0122"
seed=0
gpu_id=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --alg | -a)
            alg_name="$2"
            shift 2
            ;;
        --task | -t)
            task_name="$2"
            shift 2
            ;;
        --info | -i)
            addition_info="$2"
            shift 2
            ;;
        --seed | -s)
            seed="$2"
            shift 2
            ;;
        --gpu | -g)
            gpu_id="$2"
            shift 2
            ;;
        --debug | -d)
            DEBUG=True
            shift 1
            ;;
        --zarr_path | -z)
            zarr_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

config_name=${alg_name}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

extra_args=""
if [ -n "$zarr_path" ]; then
    extra_args="task.dataset.zarr_path=$zarr_path"
fi

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${extra_args}



                                