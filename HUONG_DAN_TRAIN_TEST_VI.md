# Huong dan train va test model cho `turtlebot3_drl`

Tai lieu nay tom tat cach train va test model cho project `turtlebot3_drl` theo dung setup Docker ma ban dang dung.

## 1. Dieu kien truoc khi chay

Ban can co san:

- Docker Desktop dang chay
- Image Docker da build xong, vi du:

```bash
docker build -t turtlebot3_auto .
```

- Workspace da `colcon build` thanh cong trong container

## 2. Cach vao dung container

Chi dung **mot** container cho toan bo qua trinh train/test.

### Lan dau mo container

```bash
docker run -it --name tb3 --privileged \
  --env DISPLAY=${DISPLAY} \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/darbar/projects/turtlebot3_auto:/home/turtlebot3_drlnav \
  --network host \
  turtlebot3_auto
```

### Nhung lan sau

Neu container `tb3` da ton tai:

```bash
docker start -ai tb3
```

### Mo them shell trong cung container

Mo terminal moi trong WSL, sau do chay:

```bash
docker exec -it tb3 bash
```

## 3. Lenh chuan can chay trong moi shell trong container

Moi khi vao container, nen chay:

```bash
cd /home/turtlebot3_drlnav
source install/setup.bash
```

Neu ban vua sua code va can build lai:

```bash
cd /home/turtlebot3_drlnav
colcon build
source install/setup.bash
```

## 4. Cach train model

Project nay can 4 terminal trong **cung mot container**.

### Terminal 1: mo Gazebo stage muon train

Vi du train o stage 4:

```bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

### Terminal 2: chay environment

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl environment
```

### Terminal 3: chay agent de train

DDPG:

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl train_agent ddpg
```

TD3:

```bash
ros2 run turtlebot3_drl train_agent td3
```

DQN:

```bash
ros2 run turtlebot3_drl train_agent dqn
```

### Terminal 4: chay node goals

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl gazebo_goals
```

## 5. Thu tu chay khuyen nghi

Trong README goc, `gazebo_goals` duoc dat truoc `environment`.

Tuy nhien voi setup Docker cua ban, thu tu sau on dinh hon:

1. `ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py`
2. `ros2 run turtlebot3_drl environment`
3. `ros2 run turtlebot3_drl train_agent ddpg`
4. `ros2 run turtlebot3_drl gazebo_goals`

Ly do: `gazebo_goals` co the publish goal dau tien qua som. Neu `environment` chua subscribe kip, agent se bi dung o trang thai:

```text
Waiting for new goal... (if persists: reset gazebo_goals node)
```

Neu gap truong hop nay, giu nguyen `launch`, `environment`, `train_agent`, sau do `Ctrl+C` terminal `gazebo_goals` va chay lai lenh:

```bash
ros2 run turtlebot3_drl gazebo_goals
```

## 6. Doi stage train

Muon train stage khac, chi can doi launch file.

Vi du stage 5:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage5.launch.py
```

Ban co the doi stage tu `1` den `10`.

## 7. Noi luu model sau khi train

Model, checkpoint, log va graph duoc luu trong:

```text
src/turtlebot3_drl/model/[HOSTNAME]/[MODEL_NAME]
```

Voi may cua ban, thuong se la:

```text
src/turtlebot3_drl/model/docker-desktop/
```

Vi du:

```text
src/turtlebot3_drl/model/docker-desktop/ddpg_0_stage_4
```

## 8. Continue training tu model da luu

Neu ban da co checkpoint va muon train tiep:

```bash
ros2 run turtlebot3_drl train_agent ddpg "ddpg_0_stage_4" 500
```

Y nghia:

- `ddpg`: thuat toan
- `"ddpg_0_stage_4"`: ten model da luu
- `500`: episode muon load lai

Chi dung lenh nay khi episode do da duoc luu that su.

## 9. Cach test model

Test model cung can mo moi truong tuong tu train, nhung doi `train_agent` thanh `test_agent`.

### Terminal 1

```bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

### Terminal 2

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl environment
```

### Terminal 3

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl test_agent ddpg "ddpg_0_stage_4" 500
```

### Terminal 4

```bash
docker exec -it tb3 bash
cd /home/turtlebot3_drlnav
source install/setup.bash
ros2 run turtlebot3_drl gazebo_goals
```

Ban cung co the test bang `td3` hoac `dqn` neu model duoc train bang cac thuat toan do.

## 10. Test model mau co san trong repo

README goc dua ra vi du test model mau nhu sau:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage9.launch.py
ros2 run turtlebot3_drl environment
ros2 run turtlebot3_drl test_agent ddpg 'examples/ddpg_0' 8000
ros2 run turtlebot3_drl gazebo_goals
```

Hoac:

```bash
ros2 run turtlebot3_drl test_agent td3 'examples/td3_0' 7400
```

## 11. Xoa model cu de train lai tu dau

Neu muon lam moi hoan toan, dung train node truoc, sau do xoa folder model cu.

Vi du:

```bash
rm -rf src/turtlebot3_drl/model/docker-desktop/ddpg_0_stage_4
rm -rf src/turtlebot3_drl/model/docker-desktop/ddpg_1_stage_4
rm -f src/turtlebot3_drl/model/docker-desktop/_ddpg_training_comparison.txt
```

Sau do chay lai quy trinh train tu dau.

## 12. Loi thuong gap

### Loi khong tim thay `/tmp/drlnav_current_stage.txt`

Nguyen nhan thuong la ban da chay cac node trong **nhieu container khac nhau**.

Khac phuc:

- chi `docker run` mot lan
- cac terminal con lai dung `docker exec -it tb3 bash`

### Loi `Waiting for new goal...`

Nguyen nhan thuong la `environment` da bo lo goal dau tien.

Khac phuc:

- chay `gazebo_goals` sau cung
- neu van bi, restart rieng node `gazebo_goals`

### Khong co GPU

May cua ban dang dung AMD, nen `torch.cuda.is_available()` co the la `False`.
Dieu nay co nghia la model dang train bang CPU, van chay duoc nhung cham hon.

## 13. Quy trinh ngan gon de dung lai moi lan

### Train tu dau

1. `docker start -ai tb3` hoac `docker run ...`
2. Terminal 1: `ros2 launch ... stage4`
3. Terminal 2: `ros2 run turtlebot3_drl environment`
4. Terminal 3: `ros2 run turtlebot3_drl train_agent ddpg`
5. Terminal 4: `ros2 run turtlebot3_drl gazebo_goals`

### Test model da luu

1. `docker start -ai tb3`
2. Terminal 1: `ros2 launch ...`
3. Terminal 2: `ros2 run turtlebot3_drl environment`
4. Terminal 3: `ros2 run turtlebot3_drl test_agent ddpg "ten_model" episode`
5. Terminal 4: `ros2 run turtlebot3_drl gazebo_goals`

---

Neu can, ban co the mo them file README goc de doi chieu:

- [README.md](/home/darbar/projects/turtlebot3_auto/README.md)
