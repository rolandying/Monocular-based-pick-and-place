# Monocular-based-pick-and-place

This Repository is for the paper: [Monocular Camera-based Robotic pick-and-place in Fusion Applications](https://www.mdpi.com/2076-3417/13/7/4487).

[![Monocular based pick and place](https://res.cloudinary.com/marcomontalbano/image/upload/v1681377973/video_to_markdown/images/youtube--z-LApEu1-hw-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=z-LApEu1-hw "Monocular based pick and place")
Dependence
----------------------------------------
Our work is depending on the [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and [rl_games](https://github.com/Denys88/rl_games).

Please first intall the [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and [rl_games](https://github.com/Denys88/rl_games) according to the introduction on their pages.
Then please copy the files in this repository to their corresponding locations strictly according to the file tree structure, replace it if a file with the same name exists. 


Training
----------------------------------------
You can start the training by the command:
```
cd IsaacGymEnvs/isaacgymenvs/
python train.py task=MyNewTask
```
Configuring your training by modifying the [MyNewTask.yaml](https://github.com/rolandying/Monocular-based-pick-and-place/blob/main/IsaacGymEnvs/isaacgymenvs/cfg/task/MyNewTask.yaml) and [MyNewTaskPPO.yaml](https://github.com/rolandying/Monocular-based-pick-and-place/blob/main/IsaacGymEnvs/isaacgymenvs/cfg/train/MyNewTaskPPO.yaml) files.


Testing
----------------------------------------
If you want to test your model, you can run the command:
```
python train.py task=MyNewTask checkpoint= the/path/to/your/model  test=TEST
```
Our model is available at [here](https://drive.google.com/file/d/1Q-O0sXF6U3BgdKJfr-EaYa_moP4s5QHz/view?usp=sharing).

The network in the code is a bit different from the description in the paper. It's even more miniaturized but the performance is about the same. 
