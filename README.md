# Monocular-based-pick-and-place

This Repository is for the paper: Monocular Camera-based Robotic pick-and-place in Fusion Applications, the link of the paper will soon be available.

Dependence
----------------------------------------
Our work is depending on the [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and [rl_games](https://github.com/Denys88/rl_games).

Please first intall the [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and [rl_games](https://github.com/Denys88/rl_games) according to the introduction on their pages.
Then please copy the files in this repository to their corresponding locations strictly according to the file tree structure, replace it if a file with the same name exists. This is only a stop-gap measure, and we will strip our code so that it can be installed and run independently as soon as possible.

Training
----------------------------------------
You can start the training by the command:
```
cd IsaacGymEnvs/isaacgymenvs/
python train.py task=MyNewTask
```
Configuring your training by modifying the [MyNewTask.yaml](https://github.com/rolandying/Monocular-based-pick-and-place/blob/main/IsaacGymEnvs/isaacgymenvs/cfg/task/MyNewTask.yaml) and [MyNewTaskPPO.yaml](https://github.com/rolandying/Monocular-based-pick-and-place/blob/main/IsaacGymEnvs/isaacgymenvs/cfg/train/MyNewTaskPPO.yaml) files


Testing
----------------------------------------
If you want to test your model, you can run the command:
```
python train.py task=MyNewTask checkpoint= the/path/to/your/model  test=TEST
```

`Under construction...`
